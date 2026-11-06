"""Lesson 4: RMSNorm, residual connections and the SwiGLU MLP -> one full Qwen3 decoder layer."""
import os, torch, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B-official")
H, H_KV, D, EPS, BASE = 16, 8, 128, 1e-6, 1e6
w = {k: t.float() for k, t in load_file(os.path.join(MODEL_DIR, "model.safetensors")).items()}
ids = AutoTokenizer.from_pretrained(MODEL_DIR)("The capital of France is", return_tensors="pt").input_ids[0]
N = len(ids)
positions = torch.arange(N)


def rms_norm(x, weight, eps=EPS):
    # Normalize over the last dim, so it works for [N, 1024] and per head on [N, heads, 128]
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


# --- Step 0: reusable functions from lesson 3 ---
def rope(x, positions):
    # x: [N, heads, 128], positions: [N] -> same shape as x
    inv_freq = 1.0 / (BASE ** (torch.arange(0, D, 2).float() / D))  # [64]
    angles = positions[:, None].float() * inv_freq[None, :]          # [N, 64]
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)  # [N, 1, 64], broadcast over heads
    x1, x2 = x.chunk(2, dim=-1)                                      # half-split pairing (j, j+64)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def attention(x, p):
    # x: already-normalized input [N, 1024]; p: layer prefix like "model.layers.0." -> [N, 1024]
    a = p + "self_attn."

    # Fused QKV: one matmul, then split and view straight into heads (no flat q/k/v kept around)
    w_qkv = torch.cat([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]])
    q, k, v = (x @ w_qkv.T).split([H * D, H_KV * D, H_KV * D], dim=-1)
    q, k, v = q.view(N, H, D), k.view(N, H_KV, D), v.view(N, H_KV, D)

    # QK-norm (per head), then RoPE on q and k only
    q = rope(rms_norm(q, w[a + "q_norm.weight"]), positions)
    k = rope(rms_norm(k, w[a + "k_norm.weight"]), positions)

    # GQA: q head h uses kv head h // 2
    k = k.repeat_interleave(H // H_KV, dim=1)
    v = v.repeat_interleave(H // H_KV, dim=1)

    # Causal scaled dot-product attention in [heads, N, 128] layout
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = q @ k.transpose(-1, -2) * D ** -0.5                      # [16, N, N]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    o = scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ v   # [16, N, 128]

    # Merge heads and project back to the residual stream width
    return o.transpose(0, 1).reshape(N, H * D) @ w[a + "o_proj.weight"].T


# --- Step 1: SwiGLU MLP ---
def mlp(x, p):
    # x: already-normalized input [N, 1024]; p: layer prefix -> [N, 1024]. Each token is processed alone.
    m = p + "mlp."

    # Fused gate+up: one matmul, then split (same trick as fused QKV / nano-vllm's gate_up_proj)
    w_gate_up = torch.cat([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]])
    gate, up = (x @ w_gate_up.T).chunk(2, dim=-1)                     # [N, 3072] each

    # SwiGLU: silu(gate) decides how much of each feature passes, up carries its value
    return (F.silu(gate) * up) @ w[m + "down_proj.weight"].T          # [N, 3072] -> [N, 1024]


# --- Step 2: one full decoder layer (pre-norm + residual) ---
def decoder_layer(h, p):
    h = h + attention(rms_norm(h, w[p + "input_layernorm.weight"]), p)
    h = h + mlp(rms_norm(h, w[p + "post_attention_layernorm.weight"]), p)
    return h


h0 = w["model.embed_tokens.weight"][ids]  # [N, 1024], the residual stream entering layer 0
mine = decoder_layer(h0, "model.layers.0.")
print("layer-0 output shape:", list(mine.shape))

# --- Step 3: compare with Hugging Face's full layer 0 ---
hf = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
cap = {}
hf.model.layers[0].register_forward_hook(lambda m, inp, o: cap.update(o=o[0] if isinstance(o, tuple) else o))
with torch.no_grad():
    hf(ids[None])
print("layer-0 max diff vs HF:", (mine - cap["o"][0]).abs().max().item())

# --- Step 4: nano-vllm's residual trick ---
# Mirrors qwen3.py:157-170: the add is delayed and done inside the NEXT norm (one fused kernel in nano-vllm).
def add_rms_norm(x, residual, weight):
    x = x + residual               # finish the delayed add -> this is the stream now
    return rms_norm(x, weight), x  # (normalized input for the sublayer, new residual)


def decoder_layer_nano(hidden, residual, p):
    # hidden: previous sublayer output (not yet added); residual: the stream before that add
    if residual is None:           # first layer: nothing to add yet
        hidden, residual = rms_norm(hidden, w[p + "input_layernorm.weight"]), hidden
    else:
        hidden, residual = add_rms_norm(hidden, residual, w[p + "input_layernorm.weight"])
    hidden = attention(hidden, p)
    hidden, residual = add_rms_norm(hidden, residual, w[p + "post_attention_layernorm.weight"])
    return mlp(hidden, p), residual  # NOT added: the next layer (or the final norm) does it


# Two layers, so the hand-off between layers (the else branch) is actually tested
L0, L1 = "model.layers.0.", "model.layers.1."
std = decoder_layer(decoder_layer(h0, L0), L1)
hidden, residual = decoder_layer_nano(h0, None, L0)
hidden, residual = decoder_layer_nano(hidden, residual, L1)
print("nano vs standard (2 layers):", (hidden + residual - std).abs().max().item())

# --- Step 5 (experiment): how big is each update compared to the residual stream? ---
attn_out = attention(rms_norm(h0, w[L0 + "input_layernorm.weight"]), L0)
h_mid = h0 + attn_out
mlp_out = mlp(rms_norm(h_mid, w[L0 + "post_attention_layernorm.weight"]), L0)
print("per-token norms (layer 0):")
print("  embedding h0 :", h0.norm(dim=-1))
print("  attention out:", attn_out.norm(dim=-1))
print("  mlp out      :", mlp_out.norm(dim=-1))
print("  layer-0 out  :", (h_mid + mlp_out).norm(dim=-1))
