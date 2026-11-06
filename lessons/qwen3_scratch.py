"""Qwen3 written from scratch in plain PyTorch (lessons 1-5), packaged for reuse.

    import qwen3_scratch as m
    m.load(dtype=torch.float32)
    tokens, caches = m.generate(prompt_ids, steps=20)
"""
import os, torch, torch.nn.functional as F
from safetensors.torch import load_file

MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B-official")
H, H_KV, D, EPS, BASE, LAYERS = 16, 8, 128, 1e-6, 1e6, 28
dev = "cuda"
w = {}  # filled by load()


def load(dtype=torch.float32):
    w.clear()
    w.update({k: t.to(dev, dtype) for k, t in load_file(os.path.join(MODEL_DIR, "model.safetensors")).items()})


def rms_norm(x, weight, eps=EPS):
    # Computed in fp32 like nano-vllm's RMSNorm, then cast back
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype) * weight


def rope(x, positions):
    inv_freq = 1.0 / (BASE ** (torch.arange(0, D, 2, device=dev).float() / D))
    angles = positions[:, None].float() * inv_freq[None, :]
    cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def mlp(x, p):
    m = p + "mlp."
    w_gate_up = torch.cat([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]])
    gate, up = (x @ w_gate_up.T).chunk(2, dim=-1)
    return (F.silu(gate) * up) @ w[m + "down_proj.weight"].T


def attention(x, p, positions, cache=None):
    a = p + "self_attn."
    n = x.shape[0]
    w_qkv = torch.cat([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]])
    q, k, v = (x @ w_qkv.T).split([H * D, H_KV * D, H_KV * D], dim=-1)
    q, k, v = q.view(n, H, D), k.view(n, H_KV, D), v.view(n, H_KV, D)
    q = rope(rms_norm(q, w[a + "q_norm.weight"]), positions)
    k = rope(rms_norm(k, w[a + "k_norm.weight"]), positions)
    if cache is not None:
        cache["k"] = torch.cat([cache["k"], k]) if "k" in cache else k
        cache["v"] = torch.cat([cache["v"], v]) if "v" in cache else v
        k, v = cache["k"], cache["v"]
    t = k.shape[0]
    k, v = k.repeat_interleave(H // H_KV, dim=1), v.repeat_interleave(H // H_KV, dim=1)
    q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)).float() * D ** -0.5            # softmax in fp32
    mask = torch.ones(n, t, dtype=torch.bool, device=dev).triu(diagonal=t - n + 1)
    o = scores.masked_fill(mask, float("-inf")).softmax(dim=-1).to(v.dtype) @ v
    return o.transpose(0, 1).reshape(n, H * D) @ w[a + "o_proj.weight"].T


def forward(ids, positions, caches=None):
    """Returns the LAST token's logits (fp32), like nano-vllm's lm_head."""
    h = w["model.embed_tokens.weight"][ids]
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        cache = None if caches is None else caches[i]
        h = h + attention(rms_norm(h, w[p + "input_layernorm.weight"]), p, positions, cache)
        h = h + mlp(rms_norm(h, w[p + "post_attention_layernorm.weight"]), p)
    return (rms_norm(h[-1], w["model.norm.weight"]) @ w["model.embed_tokens.weight"].T).float()


@torch.no_grad()
def step_logits(prompt, forced_tokens):
    """Teacher forcing: logits before each generated token, feeding `forced_tokens` instead of our own picks."""
    caches = [dict() for _ in range(LAYERS)]
    out = [forward(torch.tensor(prompt, device=dev), torch.arange(len(prompt), device=dev), caches)]
    for i, t in enumerate(forced_tokens[:-1]):
        out.append(forward(torch.tensor([t], device=dev), torch.tensor([len(prompt) + i], device=dev), caches))
    return torch.stack(out)


@torch.no_grad()
def generate(prompt, steps):
    """Greedy generation with a KV cache."""
    caches = [dict() for _ in range(LAYERS)]
    logits = forward(torch.tensor(prompt, device=dev), torch.arange(len(prompt), device=dev), caches)
    out = [int(logits.argmax())]
    for _ in range(steps - 1):
        pos = len(prompt) + len(out) - 1
        logits = forward(torch.tensor([out[-1]], device=dev), torch.tensor([pos], device=dev), caches)
        out.append(int(logits.argmax()))
    return out, caches
