"""AWQ linear layers must match dense layers loaded with the dequantized weights.

Run with: python -m tests.test_awq_linear  (from the repo root, CPU only)
"""
import os

import torch
import torch.distributed as dist

from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.quantization.awq import AWQConfig, awq_dequantize
from tests.test_awq import ref_pack

G = 128
QCFG = AWQConfig(group_size=G)


def init_dist():
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29533")
        os.environ.setdefault("USE_LIBUV", "0")
        dist.init_process_group("gloo", world_size=1, rank=0)


def random_awq(in_features, out_features):
    """Random AWQ checkpoint tensors in the on-disk layout."""
    q = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    z = torch.randint(0, 16, (in_features // G, out_features), dtype=torch.int32)
    s = (torch.rand(in_features // G, out_features) * 0.01).half()
    return {"qweight": ref_pack(q), "qzeros": ref_pack(z), "scales": s}


def dense_of(t):
    return awq_dequantize(t["qweight"], t["qzeros"], t["scales"], G, torch.float32)


def load_both(dense, awq, shards, shard_ids):
    for t, sid in zip(shards, shard_ids):
        args = () if sid is None else (sid,)
        dense.weight.weight_loader(dense.weight, dense_of(t), *args)
        for name in ("qweight", "qzeros", "scales"):
            p = getattr(awq, name)
            p.weight_loader(p, t[name], *args)


def check(dense, awq, in_features):
    x = torch.randn(3, in_features)
    torch.testing.assert_close(awq(x), dense(x), rtol=1e-4, atol=1e-4)


def test_qkv():
    hidden, head, nh, nkv = 256, 64, 4, 2
    dense = QKVParallelLinear(hidden, head, nh, nkv)
    awq = QKVParallelLinear(hidden, head, nh, nkv, quant_config=QCFG)
    shards = [random_awq(hidden, nh * head), random_awq(hidden, nkv * head), random_awq(hidden, nkv * head)]
    load_both(dense, awq, shards, ["q", "k", "v"])
    check(dense, awq, hidden)


def test_merged():
    hidden, inter = 256, 384
    dense = MergedColumnParallelLinear(hidden, [inter] * 2)
    awq = MergedColumnParallelLinear(hidden, [inter] * 2, quant_config=QCFG)
    load_both(dense, awq, [random_awq(hidden, inter), random_awq(hidden, inter)], [0, 1])
    check(dense, awq, hidden)


def test_row():
    inter, hidden = 384, 256
    dense = RowParallelLinear(inter, hidden)
    awq = RowParallelLinear(inter, hidden, quant_config=QCFG)
    load_both(dense, awq, [random_awq(inter, hidden)], [None])
    check(dense, awq, inter)


if __name__ == "__main__":
    torch.manual_seed(0)
    init_dist()
    test_qkv()
    test_merged()
    test_row()
    dist.destroy_process_group()
    print("all AWQ linear tests passed")
