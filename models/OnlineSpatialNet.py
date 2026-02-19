from typing import *

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn import Module, MultiheadAttention
from torch.nn.common_types import _size_1_t
from torch.nn.parameter import Parameter


# from models.arch.base.retention import MultiScaleRetention, RetNetRelPos


def _twoch_to_complex(two_ch: Tensor) -> Tensor:
    """(B, 2, F, T) -> (B, F, T) complex"""
    real = two_ch[:, 0]
    imag = two_ch[:, 1]
    # torch.complex does not accept bfloat16
    if real.dtype == torch.bfloat16:
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


def _complex_to_twoch(cpx: Tensor) -> Tensor:
    """(B, F, T) complex -> (B, 2, F, T)"""
    return torch.stack((cpx.real, cpx.imag), dim=1)


def _complex_to_twoch_multi(cpx: Tensor) -> Tensor:
    """(B, C, F, T) complex -> (B, 2*C, F, T)"""
    return torch.cat((cpx.real, cpx.imag), dim=1)

# try:
#     from mamba_ssm import Mamba
#     from mamba_ssm.utils.generation import InferenceParams
# except:
#     Mamba = None

from mamba_ssm import Mamba
from mamba_ssm.utils.generation import InferenceParams
class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class RetNetRelPos(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, recurrent_chunk_size: int, decay: Union[int, bool, List[int], List[float]] = None):
        super().__init__()
        angle = 1.0 / (10000**torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        if decay == False:
            self.decays = [1] * num_heads
        elif isinstance(decay, Iterable):
            if isinstance(decay[0], float):
                assert decay[0] <= 1, decay
                self.decays = decay
            else:
                assert isinstance(decay[0], int) and decay[0] > 1, decay
                self.decays = [(1 - 2**(-d)) for d in decay]
        else:
            if decay is None or decay == True:
                decay = 5
            self.decays = (1 - 2**(-decay - torch.arange(num_heads, dtype=torch.float))).tolist()
        decay = torch.log(torch.tensor(self.decays, dtype=torch.float))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = recurrent_chunk_size

    def forward(self, slen: int, activate_recurrent: bool = False, chunkwise_recurrent: bool = False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            retention_rel_pos = ((sin, cos), self.decay.exp())
        elif chunkwise_recurrent:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])

            block_index = torch.arange(self.recurrent_chunk_size).to(self.decay)
            mask = torch.tril(torch.ones(self.recurrent_chunk_size, self.recurrent_chunk_size).to(self.decay))
            mask = torch.masked_fill(block_index[:, None] - block_index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)

            value_inner_decay = mask[:, -1] / mask[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = mask.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = mask / scale

            cross_decay = torch.exp(self.decay * self.recurrent_chunk_size)
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (scale / mask[:, -1].sum(dim=-1)[:, None, None])
            cross_decay = cross_decay[:, None, None]
            retention_rel_pos = ((sin, cos), (inner_mask, cross_decay, query_inner_decay, value_inner_decay))
        else:
            index = torch.arange(slen).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            cos = torch.cos(index[:, None] * self.angle[None, :])
            mask = torch.tril(torch.ones(slen, slen).to(self.decay))
            mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
            mask = torch.exp(mask * self.decay[:, None, None])
            mask = torch.nan_to_num(mask)
            mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
            retention_rel_pos = ((sin, cos), mask)

        return retention_rel_pos

    def extra_repr(self) -> str:
        efflen = [1 / (1 - d) for d in self.decays]  # 等比数列求和
        return f"decays={self.decays} -> effective len={efflen}"

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        return


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def theta_shift(x, sin, cos) -> Tensor:
    slen = x.shape[-2]
    return (x * cos[:slen]) + (rotate_every_two(x) * sin[:slen])


def get_activation_fn(activation):
    if activation == "swish":
        return F.silu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class MultiScaleRetention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 2, gate_fn: str = "swish", look_ahead: int = 0, share_qk: bool = False):
        super().__init__()
        value_dim = embed_dim * value_factor
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = self.value_dim // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5
        self.look_ahead = look_ahead
        self.share_qk = share_qk

        self.gate_fn = get_activation_fn(activation=str(gate_fn))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False) if share_qk == False else None
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)

        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)

        self.group_norm = RMSNorm(self.head_dim, eps=1e-6, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2**-2.5) if self.share_qk == False else None
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.g_proj.weight, gain=2**-2.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=2**-1)

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, embed_dim = v.size()

        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)

        qk_mat = qr @ kr.transpose(-1, -2)  # bsz * m * tgt_len * tgt_len
        qk_mat = qk_mat * mask
        # invariant after normalization
        qk_mat = qk_mat / qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        output = torch.matmul(qk_mat, vr)
        output = output.transpose(1, 2)
        return output

    def recurrent_forward(self, qr: Tensor, kr: Tensor, v: Tensor, decay: Tensor, incremental_state: Dict[str, Any]):
        bsz = v.size(0)

        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v  # [bsz, nhead, head_dim, head_dim]
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            kv = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(self.num_heads, 1, 1) + kv / scale.sqrt().view(self.num_heads, 1, 1)
            # kv = prev_kv * decay.view(self.num_heads, 1, 1) + kv
        else:
            scale = torch.ones_like(decay)

        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale

        output = torch.sum(qr * kv, dim=3)
        return output

    def chunk_recurrent_forward(self, qr: Tensor, kr: Tensor, v: Tensor, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, embed_dim = v.size()
        chunk_len = mask.size(1)

        tgt_len0 = tgt_len
        if tgt_len % chunk_len != 0:
            qr = torch.nn.functional.pad(qr, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            kr = torch.nn.functional.pad(kr, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            v = torch.nn.functional.pad(v, pad=(0, 0, 0, chunk_len - (tgt_len % chunk_len)))
            bsz, tgt_len, embed_dim = v.size()

        num_chunks = tgt_len // chunk_len

        assert tgt_len % chunk_len == 0

        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)

        qk_mat = qr @ kr_t  # bsz * num_chunks * num_heads * chunk_len * chunk_len
        qk_mat = qk_mat * mask
        inner_scale = qk_mat.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        qk_mat = qk_mat / inner_scale
        inner_output = torch.matmul(qk_mat, v)  # bsz * num_chunks * num_heads * num_value_heads * chunk_len * head_dim

        # reduce kv in one chunk
        kv = kr_t @ (v * value_inner_decay)

        kv_recurrent = []
        cross_scale = []
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)

        # accumulate kv by loop
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = kv_state.detach().abs().sum(dim=-2, keepdim=True).max(dim=-1, keepdim=True).values.clamp(min=1)

        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)

        all_scale = torch.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (qr * query_inner_decay) @ kv_recurrent
        output = inner_output / align_inner_scale + cross_output / align_cross_scale
        # output = inner_output / cross_scale + cross_output / inner_scale

        output = output.transpose(2, 3)
        if tgt_len0 != tgt_len:
            output = output.reshape(bsz, num_chunks * chunk_len, self.num_heads, self.head_dim)
            output = output[:, :tgt_len0]

        return output

    def forward(self, x: Tensor, rel_pos: Tensor, chunkwise_recurrent: bool = False, incremental_state: Dict[str, Any] = None, rope: bool = True) -> Tensor:
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos

        q = self.q_proj(x)
        k = self.k_proj(x) if self.share_qk == False else None
        v = self.v_proj(x)
        g = self.g_proj(x)

        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        if self.share_qk == False:
            k *= self.scaling
            k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        else:
            k = q

        qr = theta_shift(q, sin, cos) if rope else q
        kr = theta_shift(k, sin, cos) if rope else k

        if self.look_ahead > 0:
            assert incremental_state is None, "not implemented for recurrent_forward"  # recurrent_forward
            # for kr, v, pad zeros at right side; for qr, pad zeros at left side
            kr = F.pad(kr, pad=(0, 0, 0, self.look_ahead))
            v = F.pad(v, pad=(0, 0, 0, self.look_ahead))
            qr = F.pad(qr, pad=(0, 0, self.look_ahead, 0))

        if incremental_state is not None:
            output = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            output = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            output = self.parallel_forward(qr, kr, v, inner_mask)

        if self.look_ahead > 0:
            output = output[:, :-self.look_ahead]

        output = self.group_norm(output).reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        output = self.gate_fn(g) * output

        output = self.out_proj(output)

        return output

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, share_qk={self.share_qk}" + (f", look_ahead={self.look_ahead}" if self.look_ahead > 0 else "")
class LinearGroup(nn.Module):

    def __init__(self, in_features: int, out_features: int, num_groups: int, bias: bool = True) -> None:
        super(LinearGroup, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight = Parameter(torch.empty((num_groups, out_features, in_features)))
        if bias:
            self.bias = Parameter(torch.empty(num_groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # same as linear
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """shape [..., group, feature]"""
        x = torch.einsum("...gh,gkh->...gk", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"{self.in_features}, {self.out_features}, num_groups={self.num_groups}, bias={True if self.bias is not None else False}"


class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o


class GlobalLayerNorm(nn.Module):
    """gLN in convtasnet"""

    def __init__(self, dim_hidden: int, seq_last: bool, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_last = seq_last
        self.eps = eps

        if seq_last:
            self.weight = Parameter(torch.empty([dim_hidden, 1]))
            self.bias = Parameter(torch.empty([dim_hidden, 1]))
        else:
            self.weight = Parameter(torch.empty([dim_hidden]))
            self.bias = Parameter(torch.empty([dim_hidden]))
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): shape [B, Seq, H] or [B, H, Seq]
        """
        var, mean = torch.var_mean(input, dim=(1, 2), unbiased=False, keepdim=True)

        output = (input - mean) / torch.sqrt(var + self.eps)
        output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, seq_last={seq_last}, eps={eps}'.format(**self.__dict__)


class BatchNorm1d(nn.Module):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__()
        self.seq_last = seq_last
        self.bn = nn.BatchNorm1d(**kwargs)

    def forward(self, input: Tensor) -> Tensor:
        if not self.seq_last:
            input = input.transpose(-1, -2)  # [B, Seq, H] -> [B, H, Seq]
        o = self.bn.forward(input)  # accepts [B, H, Seq]
        if not self.seq_last:
            o = o.transpose(-1, -2)
        return o


class GroupNorm(nn.GroupNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last == False:
            input = input.transpose(-1, 1)  # [B, ..., H] -> [B, H, ...]
        o = super().forward(input)  # accepts [B, H, ...]
        if self.seq_last == False:
            o = o.transpose(-1, 1)
        return o


class GroupBatchNorm(Module):
    """Applies Group Batch Normalization over a group of inputs

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    see: `Changsheng Quan, Xiaofei Li. NBC2: Multichannel Speech Separation with Revised Narrow-band Conformer. arXiv:2212.02076.`

    """

    dim_hidden: int
    group_size: int
    eps: float
    affine: bool
    seq_last: bool
    share_along_sequence_dim: bool

    def __init__(
        self,
        dim_hidden: int,
        group_size: Optional[int],
        share_along_sequence_dim: bool = False,
        seq_last: bool = False,
        affine: bool = True,
        eps: float = 1e-5,
        dims_norm: List[int] = None,
        dim_affine: int = None,
    ) -> None:
        """
        Args:
            dim_hidden (int): hidden dimension
            group_size (int): the size of group, optional
            share_along_sequence_dim (bool): share statistics along the sequence dimension. Defaults to False.
            seq_last (bool): whether the shape of input is [B, Seq, H] or [B, H, Seq]. Defaults to False, i.e. [B, Seq, H].
            affine (bool): affine transformation. Defaults to True.
            eps (float): Defaults to 1e-5.
            dims_norm: the dims for normalization
            dim_affine: the dims for affine transformation
        """
        super(GroupBatchNorm, self).__init__()

        self.dim_hidden = dim_hidden
        self.group_size = group_size
        self.eps = eps
        self.affine = affine
        self.seq_last = seq_last
        self.share_along_sequence_dim = share_along_sequence_dim
        if self.affine:
            if seq_last:
                weight = torch.empty([dim_hidden, 1])
                bias = torch.empty([dim_hidden, 1])
            else:
                self.weight = torch.empty([dim_hidden])
                self.bias = torch.empty([dim_hidden])

        assert (dims_norm is not None and dim_affine is not None) or (dims_norm is not None), (dims_norm, dim_affine, 'should be none at the time')
        self.dims_norm, self.dim_affine = dims_norm, dim_affine
        if dim_affine is not None:
            assert dim_affine < 0, dim_affine
            weight = weight.squeeze()
            bias = bias.squeeze()
            while dim_affine < -1:
                weight = weight.unsqueeze(-1)
                bias = bias.unsqueeze(-1)
                dim_affine += 1

        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x: Tensor, group_size: int = None) -> Tensor:
        """
        Args:
            x: shape [B, Seq, H] if seq_last=False, else shape [B, H, Seq] , where B = num of groups * group size.
            group_size: the size of one group. if not given anywhere, the input must be 4-dim tensor with shape [B, group_size, Seq, H] or [B, group_size, H, Seq]
        """
        if self.group_size != None:
            assert group_size == None or group_size == self.group_size, (group_size, self.group_size)
            group_size = self.group_size

        if group_size is not None:
            assert (x.shape[0] // group_size) * group_size, f'batch size {x.shape[0]} is not divisible by group size {group_size}'

        original_shape = x.shape
        if self.dims_norm is not None:
            var, mean = torch.var_mean(x, dim=self.dims_norm, unbiased=False, keepdim=True)
            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias
        elif self.seq_last == False:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, Seq, H = x.shape
            else:
                B, Seq, H = x.shape
                x = x.reshape(B // group_size, group_size, Seq, H)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 3), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)
        else:
            if x.ndim == 4:
                assert group_size is None or group_size == x.shape[1], (group_size, x.shape)
                B, group_size, H, Seq = x.shape
            else:
                B, H, Seq = x.shape
                x = x.reshape(B // group_size, group_size, H, Seq)

            if self.share_along_sequence_dim:
                var, mean = torch.var_mean(x, dim=(1, 2, 3), unbiased=False, keepdim=True)
            else:
                var, mean = torch.var_mean(x, dim=(1, 2), unbiased=False, keepdim=True)

            output = (x - mean) / torch.sqrt(var + self.eps)
            if self.affine:
                output = output * self.weight + self.bias

            output = output.reshape(original_shape)

        return output

    def extra_repr(self) -> str:
        return '{dim_hidden}, {group_size}, share_along_sequence_dim={share_along_sequence_dim}, seq_last={seq_last}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


def new_norm(norm_type: str, dim_hidden: int, seq_last: bool, group_size: int = None, num_groups: int = None, dims_norm: List[int] = None, dim_affine: int = None) -> nn.Module:
    if norm_type.upper() == 'LN':
        norm = LayerNorm(normalized_shape=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GBN':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=False, dims_norm=dims_norm, dim_affine=dim_affine)
    elif norm_type == 'GBNShare':
        norm = GroupBatchNorm(dim_hidden=dim_hidden, seq_last=seq_last, group_size=group_size, share_along_sequence_dim=True, dims_norm=dims_norm, dim_affine=dim_affine)
    elif norm_type.upper() == 'BN':
        norm = BatchNorm1d(num_features=dim_hidden, seq_last=seq_last)
    elif norm_type.upper() == 'GN':
        norm = GroupNorm(num_groups=num_groups, num_channels=dim_hidden, seq_last=seq_last)
    elif norm == 'gLN':
        norm = GlobalLayerNorm(dim_hidden, seq_last=seq_last)
    else:
        raise Exception(norm_type)
    return norm
class CausalConv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[_size_1_t, str] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None,
        look_ahead: int = 0,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.look_ahead = look_ahead
        assert look_ahead <= self.kernel_size[0] - 1, (look_ahead, self.kernel_size)

    def forward(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        # x [B,H,T]
        # state[name]存在，说明可以使用state里面的内容进行padding
        B, H, T = x.shape
        if state is None or id(self) not in state:
            x = F.pad(x, pad=(self.kernel_size[0] - 1 - self.look_ahead, self.look_ahead))
        else:
            x = torch.concat([state[id(self)], x], dim=-1)
        if state is not None:
            state[id(self)] = x[..., -self.kernel_size + 1:]
        x = super().forward(x)
        return x

    def extra_repr(self):
        if self.look_ahead == 0:
            return super().extra_repr()
        else:
            return super().extra_repr() + f", look ahead={self.look_ahead}"


class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
            padding: str = 'zeros',
            full: nn.Module = None,
            attention: str = 'mhsa',
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        if attention.startswith('ret'):  # e.g. ret(1,share_qk)
            attn_params = attention[4:-1].split(',')
            assert attn_params[1] in ['share_qk', 'not_share_qk'], attn_params
            value_factor = int(attn_params[0])
            self.mhsa = MultiScaleRetention(embed_dim=dim_hidden, num_heads=num_heads, value_factor=value_factor, share_qk=attn_params[1] == 'share_qk')
        elif attention.startswith('mamba'):  # e.g. mamba(16,4)
            attn_params = attention[6:-1].split(',')
            d_state, mamba_conv_kernel = int(attn_params[0]), int(attn_params[1])
            self.mhsa = Mamba(d_model=dim_hidden, d_state=d_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.attention = attention
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        if attention.startswith('mamba') and 'not_replace_ffn' not in attention:
            self.norm_tconvffn = new_norm(norms[1], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
            self.tconvffn = Mamba(d_model=dim_hidden, d_state=d_state, d_conv=mamba_conv_kernel, layer_idx=0)
        else:
            self.tconvffn = nn.ModuleList([
                new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
                nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
                nn.SiLU(),
                CausalConv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, groups=t_conv_groups),
                nn.SiLU(),
                nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
            ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None, chunkwise_recurrent: bool = True, rope: bool = True, state: Dict[int, Any] = None, inference: bool = False) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        attn = None
        if Mamba is not None and isinstance(self.mhsa, Mamba):
            x = x + self._mamba(x, self.mhsa, self.norm_mhsa, self.dropout_mhsa, inference)
        else:
            x_, attn = self._tsa(x, att_mask, chunkwise_recurrent, rope, state=state, inference=inference)
            x = x + x_
        if Mamba is not None and isinstance(self.tconvffn, Mamba):
            x = x + self._mamba(x, self.tconvffn, self.norm_tconvffn, self.dropout_tconvffn, inference)
        else:
            x = x + self._tconvffn(x, state=state)
        return x, attn

    def _mamba(self, x: Tensor, mamba: Mamba, norm: nn.Module, dropout: nn.Module, inference: bool = False):
        B, F, T, H = x.shape
        x = norm(x)
        x = x.reshape(B * F, T, H)
        if inference:
            inference_params = InferenceParams(T, B * F)
            xs = []
            for i in range(T):
                inference_params.seqlen_offset = i
                xi = mamba.forward(x[:, [i], :], inference_params)
                xs.append(xi)
            x = torch.concat(xs, dim=1)
        else:
            x = mamba.forward(x)
        x = x.reshape(B, F, T, H)
        return dropout(x)

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor], chunkwise_recurrent: bool, rope: bool = True, state: Dict[int, Any] = None, inference: bool = False) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        if isinstance(self.mhsa, MultiheadAttention):
            need_weights = False if hasattr(self, "need_weights") else self.need_weights
            # seems MHSA for long utterance inference has this issue https://github.com/pytorch/pytorch/issues/120790
            x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask, is_causal=True)
        else:
            if inference == False:
                x = self.mhsa.forward(x, rel_pos=attn_mask, incremental_state=state, chunkwise_recurrent=chunkwise_recurrent, rope=rope)
            else:
                xs, state = [], dict()
                for i in range(T):
                    xi = self.mhsa.forward(x[:, [i], :], rel_pos=attn_mask[i], incremental_state=state)
                    xs.append(xi)
                x = torch.concat(xs, dim=1)
            attn = None
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor, state: Dict[int, Any] = None) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if isinstance(m, CausalConv1d):
                x = m(x, state=state)
            elif isinstance(m, nn.GroupNorm) or "GroupNorm" in type(m).__name__:  # normalize along H & F
                x = x.reshape(B, F, -1, T).transpose(1, -1).reshape(B * T, -1, F)
                x = m(x)
                x = x.reshape(B, T, -1, F).transpose(1, -1).reshape(B * F, -1, T)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class OnlineSpatialNet(nn.Module):

    def __init__(
        self,
        dim_input: int,  # the input dim for each time-frequency point
        dim_output: int,  # the output dim for each time-frequency point
        num_layers: int,
        dim_squeeze: int,
        num_freqs: int,
        encoder_kernel_size: int = 5,
        dim_hidden: int = 192,
        dim_ffn: int = 384,
        num_heads: int = 2,
        dropout: Tuple[float, float, float] = (0, 0, 0),
        kernel_size: Tuple[int, int] = (5, 3),
        conv_groups: Tuple[int, int] = (8, 8),
        norms: List[str] = ["LN", "LN", "GN", "LN", "LN", "LN"],
        padding: str = 'zeros',
        full_share: int = 0,  # share from layer 0
        attention: str = 'mhsa(251)',  # mhsa(frames), ret(factor)
        decay: Union[int, bool, List[int], List[float]] = 5,
        chunkwise_recurrent: bool = True,
        rope: Union[bool, str] = False,
    ):
        super().__init__()
        assert attention.startswith('mhsa') or attention.startswith('ret') or attention.startswith('mamba'), attention
        assert attention.startswith('mhsa') or attention.startswith('mamba') or attention in [
            'mhsa(inf)', 'mhsa(501)', 'mhsa(251)', 'mhsa(188)', 'mhsa(126)', 'ret(2)', 'ret(2,share_qk)', 'ret(2,not_share_qk)'
        ], attention
        assert rope in [True, False, 'ALiBi'], rope
        if attention == 'ret(2)':  # 兼容之前训练的版本，在不使用旋转位置编码的时候，共享Q/K
            attention = 'ret(2,share_qk)' if rope == False else 'ret(2,not_share_qk)'

        self.num_heads = num_heads
        self.chunkwise_recurrent = chunkwise_recurrent
        self.pos = None
        if attention.startswith('ret'):
            self.pos = RetNetRelPos(embed_dim=dim_hidden, num_heads=num_heads, recurrent_chunk_size=64, decay=decay)
        elif attention.startswith('mamba'):
            self.attn_scope = 1
        else:
            import math
            self.attn_scope = int(attention[5:-1]) if attention[5:-1] != 'inf' else math.inf
        self.rope = rope

        # encoder
        self.encoder = CausalConv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, look_ahead=0)

        # spatialnet layers
        full = None
        layers = []
        for l in range(num_layers):
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
                attention=attention,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)

    def forward(self, x: Tensor, inference: bool = False, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        chunkwise_recurrent = True if inference == False else self.chunkwise_recurrent
        mask = self.get_causal_mask(slen=T, device=x.device, chunkwise_recurrent=chunkwise_recurrent, batch_size=B, inference=inference)

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for i, m in enumerate(self.layers):
            setattr(m, "need_weights", return_attn_score)
            x, attn = m(x, mask, chunkwise_recurrent, self.rope, None, inference)
            if return_attn_score:
                attns.append(attn)

        y = self.decoder(x)
        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()

    def get_causal_mask(self, slen: int, device=None, chunkwise_recurrent: bool = True, batch_size: int = None, inference: bool = False):
        if isinstance(self.pos, RetNetRelPos):
            if inference == False:
                mask = self.pos.forward(slen=slen, chunkwise_recurrent=chunkwise_recurrent)
            else:
                mask = []
                for t in range(slen):
                    rel_pos = self.pos.forward(slen=t, activate_recurrent=True)
                    mask.append(rel_pos)
        else:
            pos1 = torch.arange(start=0, end=slen, dtype=torch.long, device=device, requires_grad=False).unsqueeze(1)
            pos2 = torch.arange(start=0, end=slen, dtype=torch.long, device=device, requires_grad=False).unsqueeze(0)
            relative_pos = pos1 - pos2
            """ now, relative_pos=[
            [0,-1,-2,...,-(T-1)],
            [1, 0,-1,...,-(T-2)],
            ...
            [T-1,T-2,...,  1, 0]
            ]
            """
            if self.rope == 'ALiBi':
                assert batch_size is not None, batch_size
                m = (2.0**(-8 / torch.arange(1, self.num_heads + 1, 1, device=device))).reshape(self.num_heads, 1, 1)
                m = torch.concat([m] * batch_size, dim=0)
                relative_pos = torch.where((relative_pos >= 0) * (relative_pos < self.attn_scope), relative_pos.abs() * -1, -torch.inf)
                mask = m * relative_pos
                return mask

            mask = torch.where((relative_pos >= 0) * (relative_pos < self.attn_scope), 0.0, -torch.inf)
        return mask


class OnlineSpatialNetSeparator(nn.Module):
    """
    End-to-end separator with OnlineSpatialNet backbone:
      audio_mix (B, T) or (B, 2, T) + vec_feature (B, 6) -> separated waveforms (B, 2, T)
    """

    def __init__(self, args):
        super().__init__()
        self.stft_frame = args.network_audio["stft_frame"]
        self.stft_hop = args.network_audio["stft_hop"]
        self.n_fft = args.network_audio["n_fft"]
        self.activation = args.network_audio.get("activation", "Sigmoid")
        self.stereo_loss = args.stereo_loss
        self.register_buffer("window", torch.hann_window(self.stft_frame), persistent=False)

        input_nc = args.network_audio["input_nc"]
        if hasattr(args, "input_mono") and not args.input_mono:
            if input_nc == 2:
                input_nc = 4
        self.input_nc = input_nc
        self.output_nc = args.network_audio["output_nc"]

        self.cond_dim = args.network_audio.get("cond_dim", 0)
        self.vec_dim_per_source = args.network_audio.get("vec_dim_per_source", 3)

        if self.cond_dim > 0:
            self.DirVecNet = nn.Sequential(
                nn.Linear(self.vec_dim_per_source, self.cond_dim),
                nn.LeakyReLU(0.2, True),
                nn.Linear(self.cond_dim, self.cond_dim),
            )
        else:
            self.DirVecNet = None

        num_freqs = args.network_audio.get("num_freqs", self.n_fft // 2 + 1)
        dim_input = self.input_nc + self.cond_dim
        self.mask_net = OnlineSpatialNet(
            dim_input=dim_input,
            dim_output=self.output_nc,
            dim_squeeze=args.network_audio.get("dim_squeeze", 8),
            num_layers=args.network_audio.get("num_layers", 8),
            num_freqs=num_freqs,
            encoder_kernel_size=args.network_audio.get("encoder_kernel_size", 5),
            dim_hidden=args.network_audio.get("dim_hidden", 192),
            dim_ffn=args.network_audio.get("dim_ffn", 384),
            num_heads=args.network_audio.get("num_heads", 2),
            dropout=tuple(args.network_audio.get("dropout", (0, 0, 0))),
            kernel_size=tuple(args.network_audio.get("kernel_size", (5, 3))),
            conv_groups=tuple(args.network_audio.get("conv_groups", (8, 8))),
            norms=tuple(args.network_audio.get("norms", ("LN", "LN", "GN", "LN", "LN", "LN"))),
            padding=args.network_audio.get("padding", "zeros"),
            full_share=args.network_audio.get("full_share", 0),
            attention=args.network_audio.get("attention", "mhsa(251)"),
            decay=args.network_audio.get("decay", 5),
            chunkwise_recurrent=args.network_audio.get("chunkwise_recurrent", True),
            rope=args.network_audio.get("rope", False),
        )

    def _stft(self, audio: Tensor) -> Tensor:
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )

    def _istft(self, spec: Tensor, length: int) -> Tensor:
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )

    def _apply_activation(self, predictions: Tensor) -> Tensor:
        if self.activation == "Sigmoid":
            return torch.sigmoid(predictions)
        if self.activation == "Tanh":
            return 4 * torch.tanh(predictions)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _split_vec(self, vec_feature: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if vec_feature.dim() == 2 and vec_feature.size(1) == self.vec_dim_per_source:
            return vec_feature, None
        if vec_feature.dim() == 2 and vec_feature.size(1) == 2 * self.vec_dim_per_source:
            dir1 = vec_feature[:, : self.vec_dim_per_source]
            dir2 = vec_feature[:, self.vec_dim_per_source :]
            return dir1, dir2
        if vec_feature.dim() == 3 and vec_feature.size(1) == 2 and vec_feature.size(2) == self.vec_dim_per_source:
            return vec_feature[:, 0, :], vec_feature[:, 1, :]
        raise ValueError(
            f"vec_feature must be (B, {self.vec_dim_per_source}), (B, {2 * self.vec_dim_per_source}) "
            f"or (B, 2, {self.vec_dim_per_source}), got {tuple(vec_feature.shape)}"
        )

    def _build_input(self, mix_feat: Tensor, emb: Optional[Tensor]) -> Tensor:
        if self.cond_dim == 0:
            return mix_feat
        if emb is None:
            raise ValueError("cond_dim > 0 but emb is None")
        B, Freq, Time, _ = mix_feat.shape
        dir_map = emb[:, None, None, :].expand(B, Freq, Time, self.cond_dim)
        return torch.cat([mix_feat, dir_map], dim=-1)

    def forward(
        self,
        audio_mix: Tensor,
        vec_feature: Optional[Tensor] = None,
        return_masks: bool = False,
        inference: bool = False,
    ):
        if audio_mix.dim() == 1:
            audio_mix = audio_mix.unsqueeze(0)
        stereo_input = False
        if audio_mix.dim() == 3:
            if audio_mix.size(1) != 2:
                raise ValueError(f"stereo audio_mix must have shape (B, 2, T), got {tuple(audio_mix.shape)}")
            stereo_input = True
        elif audio_mix.dim() != 2:
            raise ValueError(f"audio_mix must be (B, T), (B, 2, T), or (T,), got {tuple(audio_mix.shape)}")
        if stereo_input and self.input_nc != 4:
            raise ValueError(f"stereo input requires input_nc=4, got input_nc={self.input_nc}")
        if not stereo_input and self.input_nc != 2:
            raise ValueError(f"mono input requires input_nc=2, got input_nc={self.input_nc}")

        B, Tlen = audio_mix.shape if not stereo_input else (audio_mix.size(0), audio_mix.size(2))

        if stereo_input:
            mix_cpx = torch.stack([self._stft(audio_mix[:, ch, :]) for ch in range(2)], dim=1)
        else:
            mix_cpx = self._stft(audio_mix)

        if stereo_input:
            mix_twoch = _complex_to_twoch_multi(mix_cpx)  # (B, 4, F, T)
        else:
            mix_twoch = _complex_to_twoch(mix_cpx)  # (B, 2, F, T)

        mix_feat = mix_twoch.permute(0, 2, 3, 1)  # (B, F, T, C)

        dir2 = None
        if self.cond_dim > 0:
            if vec_feature is None:
                raise ValueError("cond_dim > 0 but vec_feature is None")
            dir1, dir2 = self._split_vec(vec_feature)
            emb1 = self.DirVecNet(dir1) if self.DirVecNet is not None else None
            emb2 = self.DirVecNet(dir2) if (dir2 is not None and self.DirVecNet is not None) else None
        else:
            emb1 = None
            emb2 = None

        x1 = self._build_input(mix_feat, emb1)
        pred1 = self.mask_net(x1, inference=inference)
        preds = [pred1]
        if dir2 is not None:
            x2 = self._build_input(mix_feat, emb2)
            pred2 = self.mask_net(x2, inference=inference)
            preds.append(pred2)

        preds = [pred.permute(0, 3, 1, 2) for pred in preds]
        predictions = torch.stack(preds, dim=1)  # (B, num_src, output_nc, F, T)

        masks = self._apply_activation(predictions)

        sep_wavs = []
        for src_idx in range(masks.size(1)):
            mask_src = masks[:, src_idx]  # (B, output_nc, F, T)
            if stereo_input:
                if mask_src.size(1) != 4:
                    raise ValueError(
                        f"stereo mask requires output_nc=4 (L_real,L_imag,R_real,R_imag), got {mask_src.size(1)}"
                    )
                mask_L = mask_src[:, 0:2]
                mask_R = mask_src[:, 2:4]
                mask_L_cpx = _twoch_to_complex(mask_L)
                mask_R_cpx = _twoch_to_complex(mask_R)
                src_L = mix_cpx[:, 0] * mask_L_cpx
                src_R = mix_cpx[:, 1] * mask_R_cpx
                src_cpx = torch.stack([src_L, src_R], dim=1)  # (B, 2, F, T)
                src_wavs = [self._istft(src_cpx[:, ch], length=Tlen) for ch in range(2)]
                if self.stereo_loss:
                    src_wav = torch.stack(src_wavs, dim=1)  # (B, 2, T)
                else:
                    src_wav = torch.stack(src_wavs, dim=1).mean(dim=1)  # (B, T)
            else:
                if mask_src.size(1) != 2:
                    raise ValueError(f"mono mask requires output_nc=2 (real, imag), got {mask_src.size(1)}")
                mask_cpx = _twoch_to_complex(mask_src)
                src_cpx = mix_cpx * mask_cpx
                src_wav = self._istft(src_cpx, length=Tlen)
            sep_wavs.append(src_wav)

        sep_audio = torch.stack(sep_wavs, dim=1)
        masks_out = masks
        if sep_audio.size(1) == 1:
            sep_audio = sep_audio[:, 0]
            masks_out = masks[:, 0]
        if return_masks:
            return sep_audio, masks_out
        return sep_audio


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.OnlineSpatialNet
    model = OnlineSpatialNet(
        dim_input=12,
        dim_output=4,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        num_heads=4,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=["LN", "LN", "GN", "LN", "LN", "LN"],
        dim_squeeze=8,
        num_freqs=129,
        full_share=0,
        attention='mamba(16,4)',
        rope=False,
    ).cuda()
    print(model)

    x = torch.randn((1, 129, 251, 12)).cuda() # 6-channel, 4s, 8 kHz
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(model, display=False) as fcm:
        res = model(x, inference=True).mean()
        flops_forward_eval = fcm.get_total_flops()
    for k, v in fcm.get_flop_counts().items():
        ss = f"{k}: {{"
        for kk, vv in v.items():
            ss += f" {str(kk)}:{vv}"
        ss += " }"
        print(ss)
    params_eval = sum(param.numel() for param in model.parameters())
    print(f"flops_forward={flops_forward_eval/4e9:.2f}G/s, params={params_eval/1e6:.2f} M")

    # check if the implementation is causal or not
    x = torch.randn((1, 129, 1024, 12)).cuda()
    y1024 = model(x)
    y1000 = model(x[:, :, :1000, :])
    print('causal:', (y1024[:, :, :1000, :] == y1000).all().item())
