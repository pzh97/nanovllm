import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.layers.quantization.awq import AWQConfig, awq_linear


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


def set_param_attrs(param: nn.Parameter, **attrs):
    for k, v in attrs.items():
        setattr(param, k, v)


class LinearBase(nn.Module):
    """Holds either a dense weight [out, in] or AWQ tensors.

    Every parameter is tagged with:
      output_dim / input_dim: which tensor dim maps to output / input features
      pack_factor: how many logical values one element holds along output_dim
    so the tensor-parallel weight loaders below work for both layouts.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
        quant_config: AWQConfig | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.quant_config = quant_config
        if quant_config is None:
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            set_param_attrs(self.weight, output_dim=0, input_dim=1, pack_factor=1, weight_loader=self.weight_loader)
        else:
            pack, group = quant_config.pack_factor, quant_config.group_size
            assert input_size % group == 0 and output_size % pack == 0
            self.qweight = nn.Parameter(torch.empty(input_size, output_size // pack, dtype=torch.int32), requires_grad=False)
            self.qzeros = nn.Parameter(torch.empty(input_size // group, output_size // pack, dtype=torch.int32), requires_grad=False)
            self.scales = nn.Parameter(torch.empty(input_size // group, output_size, dtype=torch.float16), requires_grad=False)
            for param, pf in ((self.qweight, pack), (self.qzeros, pack), (self.scales, 1)):
                set_param_attrs(param, output_dim=1, input_dim=0, pack_factor=pf, weight_loader=self.weight_loader)
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            set_param_attrs(self.bias, output_dim=0, input_dim=None, pack_factor=1, weight_loader=self.weight_loader)
        else:
            self.register_parameter("bias", None)

    def linear(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        if self.quant_config is None:
            return F.linear(x, self.weight, bias)
        return awq_linear(x, self.qweight, self.qzeros, self.scales, bias, self.quant_config.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        super().__init__(input_size, output_size, bias, quant_config=quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        dim = param.output_dim
        shard_size = param_data.size(dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        dim = param.output_dim
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size // param.pack_factor
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size // param.pack_factor
        param_data = param_data.narrow(dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        dim = param.output_dim
        assert loaded_shard_id in ["q", "k", "v"]
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == "q":
            shard_size = q_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = kv_size
            shard_offset = q_size
        else:
            shard_size = kv_size
            shard_offset = q_size + kv_size
        shard_size //= param.pack_factor
        shard_offset //= param.pack_factor
        param_data = param_data.narrow(dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        quant_config: AWQConfig | None = None,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1, quant_config)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        dim = param.input_dim
        if dim is None:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
