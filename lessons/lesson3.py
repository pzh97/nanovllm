"""Lesson 3: Qwen3 layer-0 attention from scratch (fused QKV, QK-norm, RoPE, GQA), checked against HF."""
import os, torch, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B-official")
H, H_KV, D, EPS, BASE = 16, 8, 128, 1e-6, 1e6
w = {k: t.float() for k, t in load_file(os.path.join(MODEL_DIR, "model.safetensors")).items()}
p = "model.layers.0."
ids = AutoTokenizer.from_pretrained(MODEL_DIR)("The capital of France is", return_tensors="pt").input_ids[0]
N = len(ids)
positions = torch.arange(N)

def rms_norm(x, weight, eps=EPS):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# --- Step 1: embedding + input RMSNorm ---
embed_weight = w["model.embed_tokens.weight"]
input_embeds = embed_weight[ids]

ln_weight = w[p + "input_layernorm.weight"]
x = rms_norm(input_embeds, ln_weight, EPS)

print("step 1 complete!")
print("x shape:", list(x.shape))

# --- Step 2: fused QKV, split into heads ---
wq = w[p + "self_attn.q_proj.weight"]
wk = w[p + "self_attn.k_proj.weight"]
wv = w[p + "self_attn.v_proj.weight"]

w_qkv = torch.cat([wq, wk, wv], dim=0)
qkv = x @ w_qkv.T

q_size = H * D
kv_size = H_KV * D

q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
Q = q.view(N, H, D)
K = k.view(N, H_KV, D)
V = v.view(N, H_KV, D)
print("step 2 complete!")
print("Q shape:", list(Q.shape))
print("K shape:", list(K.shape))
print("V shape:", list(V.shape))

# --- Step 3: QK-norm (per head) ---
q_norm_weight = w[p + "self_attn.q_norm.weight"]
k_norm_weight = w[p + "self_attn.k_norm.weight"]
# Normalize per head: use the [N, heads, 128] views, not the flat [N, 2048] tensors,
# so mean(dim=-1) runs over each head's 128 dims and the [128] weight broadcasts.
Q = rms_norm(Q, q_norm_weight, EPS)
K = rms_norm(K, k_norm_weight, EPS)

print("step 3 complete!")
print("Q shape:", list(Q.shape))
print("K shape:", list(K.shape))

# --- Step 4: RoPE ---
inv_freq = 1.0 / (BASE ** (torch.arange(0, D, 2).float() / D))
angles = positions[:, None].float() * inv_freq[None, :]

cos = torch.cos(angles).unsqueeze(1)
sin = torch.sin(angles).unsqueeze(1)

def apply_rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)
    rot_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rot_x

q = apply_rope(Q, cos, sin)
k = apply_rope(K, cos, sin)

print("step 4 complete!")
print("q shape:", list(q.shape))
print("k shape:", list(k.shape))

# RoPE relative-position test: same distance (4) -> same dot product, regardless of absolute position
qa, ka = torch.randn(1, 1, D), torch.randn(1, 1, D)
def rope_at(x, pos):
    ang = torch.tensor([pos]).float()[:, None] * inv_freq[None, :]
    return apply_rope(x, ang.cos().unsqueeze(1), ang.sin().unsqueeze(1))
print("q@7 . k@3     =", (rope_at(qa, 7) * rope_at(ka, 3)).sum().item())
print("q@107 . k@103 =", (rope_at(qa, 107) * rope_at(ka, 103)).sum().item())

# --- Step 5: GQA & Attention Calculation ---
# 1. GQA Expansion: Match KV heads to Q heads (8 KV heads -> 16 Q heads, factor of 2)
n_rep = H // H_KV
k_gqa = torch.repeat_interleave(k, n_rep, dim=1)  # [N, 8, 128] -> [N, 16, 128]
v_gqa = torch.repeat_interleave(V, n_rep, dim=1)  # use per-head V [N, 8, 128], not flat v [N, 1024]

# 2. Transpose to [num_heads, seq_len, head_dim] for matrix multiplication
# q, k_gqa, v_gqa are currently [N, 16, 128] -> transpose to [16, N, 128]
q_t = q.transpose(0, 1)
k_t = k_gqa.transpose(0, 1)
v_t = v_gqa.transpose(0, 1)

# 3. Compute attention scores: [16, N, N]
# (Q @ K^T) / sqrt(head_dim)
scores = torch.matmul(q_t, k_t.transpose(-1, -2)) / (D ** 0.5)

# 4. Apply Causal Mask (prevent attending to future tokens)
seq_len = q.shape[0]
mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
scores = scores + mask

# 5. Softmax to get probabilities
attn_weights = torch.softmax(scores, dim=-1)

# 6. Multiply by V to get the final attention output: [16, N, 128]
attn_out = torch.matmul(attn_weights, v_t)

# 7. Transpose back to [N, 16, 128] and flatten heads to [N, 2048]
attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, H * D)

print("Step 5 complete!")
print("Attention output shape:", list(attn_out.shape))  # [N, 2048]
# Sanity check: PyTorch's fused attention must give the same result
ref = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True).transpose(0, 1).reshape(seq_len, H * D)
print("matches F.scaled_dot_product_attention:", torch.allclose(attn_out, ref, atol=1e-5))

# --- Step 6: Output Projection ---
# 1. Pull the output projection weight for layer 0: shape [1024, 2048] ([out, in])
o_proj_weight = w[p + "self_attn.o_proj.weight"]

# 2. Project the concatenated attention output back through o_proj
# attn_out is [N, 2048], o_proj_weight.t() is [2048, 1024] -> back to the 1024-wide residual stream
out = torch.matmul(attn_out, o_proj_weight.t())

print("Step 6 complete!")
print("Final attention block output shape:", list(out.shape))  # [N, 1024]

# --- Step 7: Compare with Hugging Face layer 0 self-attention ---
hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
cap = {}
hf.model.layers[0].self_attn.register_forward_hook(lambda m, inp, o: cap.update(o=o[0]))  # update() returns None, so HF output is untouched
with torch.no_grad():
    hf(ids[None])
print("max diff vs HF:", (out - cap["o"][0]).abs().max().item())
