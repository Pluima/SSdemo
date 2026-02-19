"""
modified from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py
#Author: Shengkui Zhao

"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torchinfo import summary
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torch import Tensor


EPS = 1e-8

# functions
class ConvModule(nn.Module):
    """
    Modified from Conformer convolution module to perform depth-wise convolution
    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolving kernel Default: 31
        dropout_p (float, optional): probability of dropout
    Inputs: inputs
        inputs (batch, time, dim): Tensor contains input sequences
    Outputs: outputs
        outputs (batch, time, dim): Tensor produces by conformer convolution module.
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 17, 
            expansion_factor: int = 2,
            dropout_p: float = 0.1,
    ) -> None:
        super(ConvModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.sequential(inputs).transpose(1, 2)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.
    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector
    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)

def identity(t, *args, **kwargs):
    return t

def append_dims(x, num_dims):
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# scalenorm

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g

# absolute positional encodings

class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x):
        n, device = x.shape[1], x.device
        t = torch.arange(n, device = device).type_as(self.inv_freq)
        sinu = einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim = -1)
        return emb * self.scale

class OffsetScale(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std = 0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim = -2)

class FFConvM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class GroupLinear(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        K = 4
    ):
        super().__init__()
        hidden = dim_in // 2
        self.group_conv = nn.Conv1d(dim_in, hidden, groups=dim_in//K, kernel_size=1)
        self.norm = nn.LayerNorm(hidden)
        self.linear = nn.Linear(hidden, dim_out)

    def forward(
        self,
        x,
    ):
        x1 = x.transpose(2,1)
        conv_out = self.group_conv(x1)
        x2 = self.norm(conv_out.transpose(2,1))
        x3 = self.linear(x2)
        return x3

class FFConvM_Small(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1,
        reduction = 4
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            GroupLinear(dim_in, dim_out),
            nn.SiLU(),
            ConvModule(dim_out),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FFM(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass = nn.LayerNorm,
        dropout = 0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
    def forward(
        self,
        x,
    ):
        output = self.mdl(x)
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 1.,
        causal = False,
        dropout = 0.1,
        rotary_pos_emb = None,
        norm_klass = nn.LayerNorm,
        shift_tokens = True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # positional embeddings
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        
        self.to_hidden = FFConvM(
            dim_in = dim,
            dim_out = hidden_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        self.to_qk = FFConvM(
            dim_in = dim,
            dim_out = query_key_dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )

        self.qk_offset_scale = OffsetScale(query_key_dim, heads = 4)

        self.to_out = FFConvM(
            dim_in = dim*2,
            dim_out = dim,
            norm_klass = norm_klass,
            dropout = dropout,
            )
        
        self.gateActivate=nn.Sigmoid() 

    def forward(
        self,
        x,
        *,
        mask = None
    ):

        """
        b - batch
        n - sequence length (within groups)
        g - group dimension
        d - feature dimension (keys)
        e - feature dimension (values)
        i - sequence dimension (source)
        j - sequence dimension (target)
        """

        # prenorm
        #x = self.fsmn(x)
        normed_x = x #self.norm(x)

        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        residual = x

        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim = -1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value = 0.)
            normed_x = torch.cat((x_shift, x_pass), dim = -1)

        # initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim = -1)
        qk = self.to_qk(normed_x)

        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        out = (att_u*v ) * self.gateActivate(att_v*u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask = None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.)
        
        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys, (quad_q, lin_q, quad_k, lin_k))

        # padding for groups
        padding = padding_to_multiple_of(n, g)

        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: F.pad(t, (0, 0, 0, padding), value = 0.), (quad_q, quad_k, lin_q, lin_k, v, u))

            mask = default(mask, torch.ones((b, n), device = device, dtype = torch.bool))
            mask = F.pad(mask, (0, padding), value = False)

        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(lambda t: rearrange(t, 'b (g n) d -> b g n d', n = self.group_size), (quad_q, quad_k, lin_q, lin_k, v, u))

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j = g)

        # calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g

        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)

        if exists(mask):
            attn = attn.masked_fill(~mask, 0.)

        if self.causal:
            causal_mask = torch.ones((g, g), dtype = torch.bool, device = device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim = 1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim = 1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value = 0.)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        return map(lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n], (quad_out_v+lin_out_v, quad_out_u+lin_out_u))
class UniDeepFsmn(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        out = x_per + self.conv1(y)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()

class UniDeepFsmn_dual(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dual, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim, bias=False)
        self.conv2 = nn.Conv2d(output_dim, output_dim, [lorder+lorder-1, 1], [1, 1], groups=output_dim//4, bias=False)

    def forward(self, input):

        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])
        conv1_out = x_per + self.conv1(y)
        z = F.pad(conv1_out, [0, 0, self.lorder - 1, self.lorder - 1])
        out = conv1_out + self.conv2(z)
        out1 = out.permute(0, 3, 2, 1)
        return input + out1.squeeze()

class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = lorder*2-1
        self.kernel_size = (self.twidth, 1)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels*(i+1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)            
            skip = torch.cat([out, skip], dim=1)
        return out

class UniDeepFsmn_dilated(nn.Module):

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth=depth
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        self.conv = DilatedDenseNet(depth=self.depth, lorder=lorder, in_channels=output_dim)

    def forward(self, input):
        f1 = F.relu(self.linear(input))
        p1 = self.project(f1)
        x = torch.unsqueeze(p1, 1)
        x_per = x.permute(0, 3, 2, 1)
        out = self.conv(x_per)
        out1 = out.permute(0, 3, 2, 1)

        return input + out1.squeeze()


class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization."""

    def __init__(self, *args, **kwargs):
        super(CLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.dim() != 3:
            raise RuntimeError('{} only accept 3-D tensor as input'.format(
                self.__name__))
        # [N, C, T] -> [N, T, C]
        sample = torch.transpose(sample, 1, 2)
        # LayerNorm
        sample = super().forward(sample)
        # [N, T, C] -> [N, C, T]
        sample = torch.transpose(sample, 1, 2)
        return sample
        
class Gated_FSMN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x) 
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input               
        return x

class Gated_FSMN_dilated(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        lorder,
        hidden_size
    ):
        super().__init__()
        self.to_u = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.to_v = FFConvM(
            dim_in = in_channels,
            dim_out = hidden_size,
            norm_klass = nn.LayerNorm,
            dropout = 0.1,
            )
        self.fsmn = UniDeepFsmn_dilated(in_channels, out_channels, lorder, hidden_size)

    def forward(
        self,
        x,
    ):
        input = x
        x_u = self.to_u(x)
        x_v = self.to_v(x) 
        x_u = self.fsmn(x_u)
        x = x_v * x_u + input               
        return x

class Gated_FSMN_Block(nn.Module):
    """Gated-FSMN block."""

    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):        
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input

class Gated_FSMN_Block_Dilated(nn.Module):
    """Gated-FSMN block with dilitations."""

    def __init__(self,
                 dim,
                 inner_channels = 256,
                 group_size = 256, 
                 norm_type = 'scalenorm',
                 ):
        super(Gated_FSMN_Block_Dilated, self).__init__()
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)
        #block dilated with gating
        self.gated_fsmn = Gated_FSMN_dilated(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)

    def forward(self, input):
        conv1 = self.conv1(input.transpose(2,1))
        norm1 = self.norm1(conv1)
        seq_out = self.gated_fsmn(norm1.transpose(2,1))
        norm2 = self.norm2(seq_out.transpose(2,1))
        conv2 = self.conv2(norm2)
        return conv2.transpose(2,1) + input

class MossformerBlock_GFSMN(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.fsmn = nn.ModuleList([Gated_FSMN_Block_Dilated(dim) for _ in range(depth)])
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens) for _ in range(depth)])
  
    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            x = self.fsmn[ii](x)
            ii = ii + 1
        return x

class MossformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size = 256, #384, #128, #256,
        query_key_dim = 128, #256, #128,
        expansion_factor = 4.,
        causal = False,
        attn_dropout = 0.1,
        norm_type = 'scalenorm',
        shift_tokens = True
    ):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        rotary_pos_emb = RotaryEmbedding(dim = min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.layers = nn.ModuleList([FLASH_ShareA_FFConvM(dim = dim, group_size = group_size, query_key_dim = query_key_dim, expansion_factor = expansion_factor, causal = causal, dropout = attn_dropout, rotary_pos_emb = rotary_pos_emb, norm_klass = norm_klass, shift_tokens = shift_tokens) for _ in range(depth)])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(
        self,
        x,
        *,
        mask = None
    ):
        ii = 0
        for flash in self.layers:
            x = flash(x, mask = mask)
            ii = ii + 1
        return x


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class IdentityBlock:
    """This block is used when we want to have identity transformation within the Dual_path block.

    Example
    -------
    >>> x = torch.randn(10, 100)
    >>> IB = IdentityBlock()
    >>> xhat = IB(x)
    """

    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x


class MossFormerM(nn.Module):
    """This class implements the MossFormer2 block.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

        self.mossformerM = MossformerBlock_GFSMN(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output

class MossFormerM2(nn.Module):
    """This class implements the MossFormer block.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM2(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

        self.mossformerM = MossformerBlock(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output

class Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> comp_block = Computation_Block(64)
        >>> x = torch.randn(10, 64, 100)
        >>> x = comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Computation_Block, self).__init__()

        ##MossFormer+: MossFormer with recurrence
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        ##MossFormerM2: the orignal MossFormer
        #self.intra_mdl = MossFormerM2(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, S].
            where, B = Batchsize,
               N = number of filters
               S = sequence time index 
        """
        B, N, S = x.shape
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous() #.view(B, S, N)

        intra = self.intra_mdl(intra)

        # [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x

        out = intra
        return out


class MossFormer_MaskNet(nn.Module):
    """The MossFormer MaskNet for predicting mask for encoder output features.
       The MossFormer2 model uses an upgraded MaskNet structure

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> mossformer_masknet = MossFormer_MaskNet(64, 64, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = mossformer_masknet(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=24,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer_MaskNet, self).__init__()
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = Computation_Block(
                    num_blocks,
                    out_channels,
                    norm,
                    skip_around_intra=skip_around_intra,
                )

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, S]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               S = the number of time frames
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d_encoder(x)
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(1, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(0, -1) 
            x = base + emb
            

        # [B, N, S]
        x = self.mdl(x)
        x = self.prelu(x)

        # [B, N*spks, S]
        x = self.conv1d_out(x)
        B, _, S = x.shape

        # [B*spks, N, S]
        x = x.view(B * self.num_spks, -1, S)

        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, S]
        x = self.conv1_decoder(x)

        # [B, spks, N, S]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, S]
        x = x.transpose(0, 1)

        return x

class MossFormer(nn.Module):
    """ The E2E Encoder-MaskNet-Decoder MossFormer model for speech separation
        The MossFormer2 model uses an upgraded MaskNet
    ---------
    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the MossFormer2 blocks.
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> mossformer = MossFormer(num_spks=2)
    >>> x = torch.randn(1, 10000)
    >>> x = mossformer(x)
    >>> x
    x[0]: torch.Size([1, 10000])
    x[1]: torch.Size([1, 10000])
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer, self).__init__()
        self.num_spks = num_spks
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=1)
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = Decoder(
           in_channels=out_channels,
           out_channels=1,
           kernel_size=kernel_size,
           stride = kernel_size//2,
           bias=False
        )
    def forward(self, input):
        x = self.enc(input)
        mask = self.mask_net(x)
        x = torch.stack([x] * self.num_spks)
        sep_x = x * mask

        # Decoding
        est_source = torch.cat(
            [
                self.dec(sep_x[i]).unsqueeze(-1)
                for i in range(self.num_spks)
            ],
            dim=-1,
        )
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        out = []
        for spk in range(self.num_spks):
            out.append(est_source[:,:,spk])
        return out


class MossFormer2_SS(nn.Module):
    """MossFormer2 model wrapper for outside calling"""

    def __init__(self, args):
        super(MossFormer2_SS, self).__init__()
        self.model = MossFormer(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.mossformer_sequence_dim,
            num_blocks=args.num_mossformer_layer,
            kernel_size=args.encoder_kernel_size,
            norm="ln",
            num_spks=args.num_spks,
            skip_around_intra=True,
            use_global_pos_enc=True,
            max_length=20000)

    def forward(self, x):
        outputs = self.model(x)
        return outputs
