"""Lesson 2: causal scaled dot-product attention from scratch, checked against PyTorch."""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
N, d = 4, 8  # 4 tokens, head dim 8
q, k, v = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)

print("=== Step 1: Scores ===")
scores = q @ k.T * d**-0.5
print(f"Scores shape: {list(scores.shape)}")
print(f"scores[2, 0] meaning: Token 2's raw attention affinity score toward Key 0.\n")

print("=== Step 2: Softmax ===")
w = scores.softmax(dim=-1)
print(f"w.sum(-1) is all ones: {torch.allclose(w.sum(dim=-1), torch.ones(N))}")
print("Why dim=-1? Because we want each query token to independently distribute its 100% attention budget across the key tokens (the last dimension).\n")

print("=== Step 3: Causal Mask ===")
mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
scores_masked = scores.masked_fill(mask, float("-inf"))
w_masked = scores_masked.softmax(dim=-1)
print("Causal Softmax Output w:")
print(w_masked)
print("Notice row 0 is [1, 0, 0, 0] and the matrix is lower-triangular.\n")

print("=== Step 4: Output ===")
out = w_masked @ v
print(f"Output shape: {list(out.shape)}\n")

print("=== Step 5: Check against PyTorch (Single-Head) ===")
def my_attention(q, k, v):
    d_k = q.shape[-1]
    scores = q @ k.T * d_k**-0.5
    N_seq = q.shape[0]
    mask = torch.triu(torch.ones(N_seq, N_seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    w = scores.softmax(dim=-1)
    return w @ v

ref = F.scaled_dot_product_attention(q[None], k[None], v[None], is_causal=True)
print("Single-head match check:", torch.allclose(my_attention(q, k, v), ref[0], atol=1e-6))


print("=== Step 6: Multi-Head Attention ===")
N, H, head_dim = 4, 2, 4
q_mh = torch.randn(N, H, head_dim)
k_mh = torch.randn(N, H, head_dim)
v_mh = torch.randn(N, H, head_dim)

def my_attention_multihead(q, k, v):
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    scores = q @ k.transpose(-2, -1) * (q.shape[-1] ** -0.5)

    N_seq = q.shape[1]
    mask = torch.triu(torch.ones(N_seq, N_seq, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    w = scores.softmax(dim=-1)
    out = w @ v
    return out.transpose(0, 1)

ref_mh = F.scaled_dot_product_attention(
    q_mh.transpose(0, 1),
    k_mh.transpose(0, 1),
    v_mh.transpose(0, 1),
    is_causal=True
).transpose(0, 1)

print("Multi-head match check:", torch.allclose(my_attention_multihead(q_mh, k_mh, v_mh), ref_mh, atol=1e-6))

print("=== Step 7: Causality ===")
# Replace the LAST token's k and v with noise: only the last output row may change
out_before = my_attention_multihead(q_mh, k_mh, v_mh)
k_new, v_new = k_mh.clone(), v_mh.clone()
k_new[-1], v_new[-1] = torch.randn(H, head_dim), torch.randn(H, head_dim)
out_after = my_attention_multihead(q_mh, k_new, v_new)
print("rows unchanged:", [torch.equal(out_before[i], out_after[i]) for i in range(N)])
