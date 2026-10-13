"""Run with: python -m tests.test_awq  (from the repo root)"""
import torch

from nanovllm.layers.quantization.awq import awq_unpack, awq_dequantize


def ref_pack(w: torch.Tensor) -> torch.Tensor:
    """Reference packer copied from AutoAWQ's WQLinear_GEMM.from_linear. w: [R, C] in 0..15."""
    order_map = [0, 2, 4, 6, 1, 3, 5, 7]
    R, C = w.shape
    out = torch.zeros(R, C // 8, dtype=torch.int32)
    for c in range(C // 8):
        for i in range(8):
            out[:, c] |= w[:, c * 8 + order_map[i]].int() << (4 * i)
    return out


def make_case(IN=256, OUT=64, G=128):
    torch.manual_seed(0)
    q = torch.randint(0, 16, (IN, OUT), dtype=torch.int32)
    z = torch.randint(0, 16, (IN // G, OUT), dtype=torch.int32)
    s = torch.rand(IN // G, OUT, dtype=torch.float16) * 0.01
    return q, z, s, G


def test_unpack_roundtrip():
    q, *_ = make_case()
    assert torch.equal(awq_unpack(ref_pack(q)), q), "unpack order is wrong"


def test_unpack_negative_int32():
    # 0xc8f406b9 has its top bit set, so it is negative as int32.
    packed = torch.tensor([[0xC8F406B9 - 2**32]], dtype=torch.int32)
    assert awq_unpack(packed).tolist() == [[9, 4, 11, 15, 6, 8, 0, 12]]


def test_dequantize():
    q, z, s, G = make_case()
    W = awq_dequantize(ref_pack(q), ref_pack(z), s, G)
    ref = ((q - z.repeat_interleave(G, 0)).half() * s.repeat_interleave(G, 0)).T
    assert W.shape == ref.shape and torch.allclose(W, ref), "dequantization is wrong"


if __name__ == "__main__":
    test_unpack_roundtrip()
    test_unpack_negative_int32()
    test_dequantize()
    print("all AWQ dequant tests passed")
