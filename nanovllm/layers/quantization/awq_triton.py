"""Fused AWQ dequantize + GEMM in Triton.

C[M, N] = A[M, K] @ dequant(qweight[K, N/8], qzeros[K/G, N/8], scales[K/G, N])

The int4 weight tile is unpacked and dequantized in registers right before
tl.dot, so the full fp16 weight never touches global memory. That matters
for decode, which is memory-bound: we read 4 bits per weight instead of
writing and re-reading 16.
"""
import functools

import torch
import triton
import triton.language as tl


@triton.jit
def awq_gemm_kernel(
    a_ptr, qweight_ptr, qzeros_ptr, scales_ptr, c_ptr,
    M, N, K, group_size,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_pn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)  # packed columns
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    # Logical column n sits in packed column n // 8 at bit shift
    # [0, 16, 4, 20, 8, 24, 12, 28][n % 8] (AWQ's interleaved packing order).
    shifts = (offs_n % 2) * 16 + ((offs_n % 8) // 2) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Each split-K program walks every SPLIT_K-th K tile.
    for k0 in range(pid_k * BLOCK_K, K, BLOCK_K * SPLIT_K):
        k = k0 + offs_k
        a = tl.load(a_ptr + offs_m[:, None] * K + k[None, :], mask=mask_m[:, None], other=0.0)

        # Load each packed int32 once, then repeat it 8x in registers:
        # [BLOCK_K, BLOCK_N/8] -> [BLOCK_K, BLOCK_N] with q[:, n] = packed[:, n // 8].
        q = tl.load(qweight_ptr + k[:, None] * (N // 8) + offs_pn[None, :])
        q = tl.interleave(q, q)
        q = tl.interleave(q, q)
        q = tl.interleave(q, q)
        q = (q >> shifts[None, :]) & 0xF

        # BLOCK_K divides group_size, so the whole tile shares one quantization group.
        g = k0 // group_size
        z = tl.load(qzeros_ptr + g * (N // 8) + offs_pn)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = tl.interleave(z, z)
        z = (z >> shifts) & 0xF
        s = tl.load(scales_ptr + g * N + offs_n)

        b = (q - z[None, :]).to(a.dtype) * s[None, :].to(a.dtype)
        acc += tl.dot(a, b)

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_K == 1:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask_m[:, None])
    else:
        tl.atomic_add(c_ptrs, acc, mask=mask_m[:, None])


@triton.jit
def awq_dequantize_kernel(
    qweight_ptr, qzeros_ptr, scales_ptr, out_ptr,
    K, N, group_size,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_k < K)[:, None] & (offs_n < N)[None, :]
    packed_n = offs_n // 8
    shifts = (offs_n % 2) * 16 + ((offs_n % 8) // 2) * 4
    g = offs_k // group_size

    q = tl.load(qweight_ptr + offs_k[:, None] * (N // 8) + packed_n[None, :], mask=mask, other=0)
    z = tl.load(qzeros_ptr + g[:, None] * (N // 8) + packed_n[None, :], mask=mask, other=0)
    s = tl.load(scales_ptr + g[:, None] * N + offs_n[None, :], mask=mask, other=0.0)
    q = (q >> shifts[None, :]) & 0xF
    z = (z >> shifts[None, :]) & 0xF
    w = (q - z).to(tl.float32) * s.to(tl.float32)
    tl.store(out_ptr + offs_k[:, None] * N + offs_n[None, :], w.to(out_ptr.dtype.element_ty), mask=mask)


def awq_dequantize_triton(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Returns the dequantized weight as [K, N] (in, out) -- use x @ w, not F.linear."""
    K, N = qweight.shape[0], scales.shape[1]
    out = torch.empty((K, N), dtype=dtype, device=qweight.device)
    grid = (triton.cdiv(K, 64), triton.cdiv(N, 64))
    awq_dequantize_kernel[grid](qweight, qzeros, scales, out, K, N, group_size, BLOCK_K=64, BLOCK_N=64)
    return out


def _config(M: int, N: int, K: int, num_sms: int):
    """Tile sizes picked by a sweep on an RTX 4070 for decode-sized M."""
    block_m = 16 if M <= 16 else 32 if M <= 32 else 64
    block_n = 128 if N % 128 == 0 else 64
    block_k = 32
    # Too few output tiles (small M, small N) -> split K so the grid fills ~2 waves of SMs.
    tiles = triton.cdiv(M, block_m) * (N // block_n)
    split_k = 1
    while tiles * split_k < 2 * num_sms and split_k < 8 and K % (block_k * split_k * 2) == 0:
        split_k *= 2
    return block_m, block_n, block_k, split_k


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def awq_gemm_triton(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    assert x.is_contiguous() and x.dim() == 2
    M, K = x.shape
    N = scales.shape[1]
    block_m, block_n, block_k, split_k = _config(M, N, K, _num_sms(x.device))
    assert group_size % block_k == 0 and K % group_size == 0 and N % block_n == 0
    # atomic_add for bf16 is not supported everywhere, so accumulate split-K in fp32.
    out_dtype = x.dtype if split_k == 1 else torch.float32
    alloc = torch.empty if split_k == 1 else torch.zeros
    c = alloc((M, N), dtype=out_dtype, device=x.device)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n), split_k)
    awq_gemm_kernel[grid](
        x, qweight, qzeros, scales, c,
        M, N, K, group_size,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k, SPLIT_K=split_k,
        num_warps=4, num_stages=3,
    )
    return c.to(x.dtype)
