from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AWQConfig:
    group_size: int = 128
    bits: int = 4

    @property
    def pack_factor(self) -> int:
        return 32 // self.bits

    @classmethod
    def from_hf(cls, quantization_config: dict | None) -> "AWQConfig | None":
        if not quantization_config:
            return None
        method = quantization_config.get("quant_method")
        assert method == "awq", f"unsupported quantization method: {method}"
        assert quantization_config.get("version", "gemm") == "gemm", "only AWQ GEMM packing is supported"
        assert quantization_config.get("zero_point", True), "only zero-point AWQ is supported"
        assert not quantization_config.get("modules_to_not_convert"), "modules_to_not_convert is not supported"
        cfg = cls(group_size=quantization_config["group_size"], bits=quantization_config["bits"])
        assert cfg.bits == 4, "only 4-bit AWQ is supported"
        return cfg


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
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dequantize AWQ tensors into an [out, in] weight usable by F.linear.

    qweight: [in, out/8] int32, qzeros: [in/group, out/8] int32, scales: [in/group, out].
    w[i, j] = (q[i, j] - z[i // group, j]) * s[i // group, j]
    """
    dtype = dtype or scales.dtype
    q = awq_unpack(qweight)
    z = awq_unpack(qzeros).repeat_interleave(group_size, dim=0)
    s = scales.repeat_interleave(group_size, dim=0)
    w = (q - z).to(dtype) * s.to(dtype)
    return w.T.contiguous()


def awq_linear(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
) -> torch.Tensor:
    """y = x @ W.T + b with an AWQ weight.

    Reference path: materialize the full weight in x.dtype, then call F.linear.
    Correct but memory-bound; a fused kernel replaces this later.
    """
    w = awq_dequantize(qweight, qzeros, scales, group_size, x.dtype)
    return F.linear(x, w, bias)
