import torch

# AWQ (GEMM version) packs 8 int4 values into one int32 in the interleaved
# order [0, 2, 4, 6, 1, 3, 5, 7], i.e. nibble i holds column order[i].
# The inverse permutation below tells which nibble holds column j.
AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]


def awq_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Unpack [R, C/8] int32 into [R, C] int32 values in 0..15, natural column order."""
    shifts = torch.arange(0, 32, 4, device=packed.device)
    # Arithmetic right shift may drag in sign bits for negative int32s; & 0xF drops them.
    x = (packed.unsqueeze(-1) >> shifts) & 0xF  # [R, C/8, 8] in nibble order
    x = x[..., AWQ_REVERSE_ORDER]  # back to column order
    return x.reshape(packed.shape[0], -1)


def awq_dequantize(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize AWQ tensors into an [out, in] weight usable by F.linear.

    qweight: [in, out/8] int32, qzeros: [in/group, out/8] int32, scales: [in/group, out].
    w[i, j] = (q[i, j] - z[i // group, j]) * s[i // group, j]
    """
    q = awq_unpack(qweight)
    z = awq_unpack(qzeros).repeat_interleave(group_size, dim=0)
    s = scales.repeat_interleave(group_size, dim=0)
    w = (q - z).to(scales.dtype) * s
    return w.T.contiguous()
