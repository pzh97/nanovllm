"""Lesson 5: generation with a KV cache (prefill vs decode), then a toy paged KV cache."""
import os, time, torch, torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B-official")
H, H_KV, D, EPS, BASE, LAYERS = 16, 8, 128, 1e-6, 1e6, 28
dev = "cuda"
w = {k: t.float().to(dev) for k, t in load_file(os.path.join(MODEL_DIR, "model.safetensors")).items()}
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
prompt = tok("The capital of France is", return_tensors="pt").input_ids[0].tolist()
STEPS = 20


# --- Your lesson 4 building blocks (now with any length n instead of a global N) ---
def rms_norm(x, weight, eps=EPS):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rope(x, positions):
    inv_freq = 1.0 / (BASE ** (torch.arange(0, D, 2, device=dev).float() / D))
    angles = positions[:, None].float() * inv_freq[None, :]
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def mlp(x, p):
    m = p + "mlp."
    w_gate_up = torch.cat([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]])
    gate, up = (x @ w_gate_up.T).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w[m + "down_proj.weight"].T


def attention(x, p, positions, cache=None):
    # x: [n, 1024] new tokens only; positions: [n] their absolute positions
    # cache: None (no cache) or this layer's dict {"k": [T, 8, 128], "v": [T, 8, 128]}, updated in place
    a = p + "self_attn."
    n = x.shape[0]
    w_qkv = torch.cat([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]])
    q, k, v = (x @ w_qkv.T).split([H * D, H_KV * D, H_KV * D], dim=-1)
    q, k, v = q.view(n, H, D), k.view(n, H_KV, D), v.view(n, H_KV, D)
    q = rope(rms_norm(q, w[a + "q_norm.weight"]), positions)
    k = rope(rms_norm(k, w[a + "k_norm.weight"]), positions)

    if cache is not None:
        # Step 1: append the new k, v (k already rotated: a past token's position never changes)
        # and let the new queries attend over every cached token
        cache["k"] = torch.cat([cache["k"], k]) if "k" in cache else k
        cache["v"] = torch.cat([cache["v"], v]) if "v" in cache else v
        k, v = cache["k"], cache["v"]

    t = k.shape[0]  # number of keys = past + new
    k, v = k.repeat_interleave(H // H_KV, dim=1), v.repeat_interleave(H // H_KV, dim=1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = q @ k.transpose(-1, -2) * D ** -0.5                      # [16, n, t]
    # Query i is at absolute position t-n+i, so it may see keys 0..t-n+i.
    # No cache (n == t): ordinary causal mask.  Decode (n == 1): nothing masked.
    mask = torch.ones(n, t, dtype=torch.bool, device=dev).triu(diagonal=t - n + 1)
    o = scores.masked_fill(mask, float("-inf")).softmax(dim=-1) @ v
    return o.transpose(0, 1).reshape(n, H * D) @ w[a + "o_proj.weight"].T


def forward(ids, positions, caches=None):
    # Full model: embed -> 28 layers -> final norm -> lm_head (tied to the embedding). Returns the LAST token's logits.
    h = w["model.embed_tokens.weight"][ids]
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        cache = None if caches is None else caches[i]
        h = h + attention(rms_norm(h, w[p + "input_layernorm.weight"]), p, positions, cache)
        h = h + mlp(rms_norm(h, w[p + "post_attention_layernorm.weight"]), p)
    return rms_norm(h[-1], w["model.norm.weight"]) @ w["model.embed_tokens.weight"].T


# --- Generation without a cache: recompute the whole sequence every step ---
@torch.no_grad()
def generate_no_cache():
    ids = list(prompt)
    for _ in range(STEPS):
        logits = forward(torch.tensor(ids, device=dev), torch.arange(len(ids), device=dev))
        ids.append(int(logits.argmax()))  # greedy
    return ids[len(prompt):]


# --- Step 2: generation WITH a cache ---
@torch.no_grad()
def generate_with_cache():
    caches = [dict() for _ in range(LAYERS)]

    # Prefill: the whole prompt in one pass fills every layer's cache and gives the first token
    logits = forward(torch.tensor(prompt, device=dev), torch.arange(len(prompt), device=dev), caches)
    out = [int(logits.argmax())]

    # Decode: feed only the newest token, at its absolute position
    for _ in range(STEPS - 1):
        pos = len(prompt) + len(out) - 1
        logits = forward(torch.tensor([out[-1]], device=dev), torch.tensor([pos], device=dev), caches)
        out.append(int(logits.argmax()))
    return out, caches


# --- Step 3: compare speed, output and cache size ---
for fn in (generate_no_cache, generate_with_cache):
    fn()  # warmup
    torch.cuda.synchronize(); t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    print(f"{fn.__name__:20s} {time.perf_counter() - t0:.3f}s")

a = generate_no_cache()
b, caches = generate_with_cache()
print("same tokens:", a == b)
print("text:", repr(tok.decode(b)))

num_tokens = caches[0]["k"].shape[0]                                          # prompt + generated - 1
cache_bytes = sum(c["k"].numel() + c["v"].numel() for c in caches) * 2        # bf16 = 2 bytes
print(f"cache: {num_tokens} tokens, {cache_bytes // num_tokens} bytes/token in bf16 "
      f"(expected 2 * {H_KV} * {D} * 2 * {LAYERS} = {2 * H_KV * D * 2 * LAYERS})")


# --- Step 4: toy paged KV cache (what nano-vllm actually does) ---
BS, NUM_BLOCKS = 4, 8                                             # nano-vllm uses BS=256
k_pool = torch.zeros(NUM_BLOCKS * BS, H_KV * D, device=dev)       # one row per slot, all blocks
seq_len = 10
block_table = [5, 2, 7]                                           # this sequence's physical blocks
k_seq = torch.randn(seq_len, H_KV * D, device=dev)                # its keys, positions 0..9

# a) where each position lives: block_table[pos // BS] picks the block, pos % BS the offset in it
slot_mapping = [block_table[pos // BS] * BS + pos % BS for pos in range(seq_len)]
# b) write: what store_kvcache_kernel does in attention.py
k_pool[slot_mapping] = k_seq
# c) read back: walk the block table in order (what FlashAttention does with block_table)
read_back = torch.cat([k_pool[b * BS:(b + 1) * BS] for b in block_table])[:seq_len]
# d) check
print("slot_mapping:", slot_mapping)
print("paged read-back == original:", torch.equal(read_back, k_seq))
