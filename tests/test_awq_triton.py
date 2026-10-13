"""Triton AWQ kernels vs the PyTorch reference, plus a decode micro-benchmark.

Run with: python -m tests.test_awq_triton  (from the repo root, needs a GPU)
"""
import torch
import torch.nn.functional as F
import triton

from nanovllm.layers.quantization.awq import awq_dequantize, awq_linear
from nanovllm.layers.quantization.awq_triton import awq_gemm_triton, awq_dequantize_triton
from tests.test_awq import ref_pack

G = 128
# (K, N) of real projections: Qwen3-0.6B qkv / down, Qwen2.5-7B qkv / gate_up / down
SHAPES = [(1024, 4096), (3072, 1024), (3584, 4608), (3584, 37888), (18944, 3584)]


def random_awq(K, N, device="cuda"):
    q = torch.randint(0, 16, (K, N), dtype=torch.int32)
    z = torch.randint(0, 16, (K // G, N), dtype=torch.int32)
    s = (torch.rand(K // G, N) * 0.02 + 0.001).half()
    return ref_pack(q).to(device), ref_pack(z).to(device), s.to(device)


def test_dequantize():
    for K, N in SHAPES[:3]:
        qw, qz, s = random_awq(K, N)
        ref = awq_dequantize(qw, qz, s, G, torch.float16)  # [N, K]
        out = awq_dequantize_triton(qw, qz, s, G, torch.float16)  # [K, N]
        torch.testing.assert_close(out, ref.T, rtol=0, atol=0)


def test_gemm():
    for dtype in (torch.float16, torch.bfloat16):
        for K, N in SHAPES[:3]:
            qw, qz, s = random_awq(K, N)
            w = awq_dequantize(qw, qz, s, G, dtype)
            for M in (1, 5, 16, 33, 100, 256):
                x = torch.randn(M, K, device="cuda", dtype=dtype) * 0.1
                ref = F.linear(x.float(), w.float())
                out = awq_gemm_triton(x, qw, qz, s, G)
                assert out.dtype == dtype and out.shape == (M, N)
                err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
                assert err < 2e-2, f"{dtype} M={M} K={K} N={N}: rel err {err:.4f}"


def test_awq_linear_dispatch():
    K, N = 1024, 4096
    qw, qz, s = random_awq(K, N)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    w = awq_dequantize(qw, qz, s, G, torch.float16)
    for M in (8, 600):  # fused path and dequant + cuBLAS path
        x = torch.randn(M, K, device="cuda", dtype=torch.float16) * 0.1
        ref = F.linear(x.float(), w.float(), bias.float())
        out = awq_linear(x, qw, qz, s, bias, G).float()
        assert (out - ref).abs().max().item() / ref.abs().max().item() < 2e-2


def bench_decode():
    print(f"{'K x N':>14} {'M':>4} {'dense fp16':>11} {'dequant+mm':>11} {'fused':>8}  (us)")
    for K, N in SHAPES[2:]:
        qw, qz, s = random_awq(K, N)
        w = awq_dequantize(qw, qz, s, G, torch.float16)
        for M in (1, 16, 64):
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            t_dense = triton.testing.do_bench(lambda: F.linear(x, w)) * 1e3
            t_naive = triton.testing.do_bench(lambda: x @ awq_dequantize_triton(qw, qz, s, G, x.dtype)) * 1e3
            t_fused = triton.testing.do_bench(lambda: awq_gemm_triton(x, qw, qz, s, G)) * 1e3
            print(f"{K:>6} x {N:<6} {M:>4} {t_dense:>11.0f} {t_naive:>11.0f} {t_fused:>8.0f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_dequantize()
    test_gemm()
    test_awq_linear_dispatch()
    print("all AWQ Triton tests passed")
    bench_decode()
