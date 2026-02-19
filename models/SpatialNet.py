from typing import *

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch import Tensor
from torch.nn import MultiheadAttention
import math


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
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
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
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
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
        x_, attn = self._tsa(x, att_mask)
        x = x + x_
        x = x + self._tconvffn(x)
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask)
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
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


class SpatialNet(nn.Module):

    def __init__(
            self,
            dim_input: int,  # the input dim for each time-frequency point
            dim_output: int,  # the output dim for each time-frequency point
            dim_squeeze: int,
            num_layers: int,
            num_freqs: int,
            encoder_kernel_size: int = 5,
            dim_hidden: int = 192,
            dim_ffn: int = 384,
            num_heads: int = 2,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full_share: int = 0,  # share from layer 0
    ):
        super().__init__()

        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")

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
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)
    def _stft(self, audio):
        # audio: (B, T)
        return torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(audio.device),
            center=True,
            return_complex=True,
        )  # (B, F, T) complex

    def _istft(self, spec, length):
        # spec: (B, F, T) complex
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.stft_hop,
            win_length=self.stft_frame,
            window=self.window.to(spec.device),
            center=True,
            length=length,
        )  # (B, T)

    def forward(self, x: Tensor, return_attn_score: bool = False) -> Tensor:
        
        # x: [Batch, Freq, Time, Feature]
        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for m in self.layers:
            setattr(m, "need_weights", return_attn_score)
            x, attn = m(x)
            if return_attn_score:
                attns.append(attn)

        y = self.decoder(x)
        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()


class SpatialNetSeparator(nn.Module):
    """
    End-to-end separator with SpatialNet backbone:
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
        self.mask_net = SpatialNet(
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

    def _split_vec(self, vec_feature: Tensor) -> Tuple[Tensor, Tensor]:
        if vec_feature.dim() == 2 and vec_feature.size(1) == 6:
            dir1 = vec_feature[:, :3]
            dir2 = vec_feature[:, 3:]
        elif vec_feature.dim() == 3 and vec_feature.size(1) == 2 and vec_feature.size(2) == 3:
            dir1 = vec_feature[:, 0, :]
            dir2 = vec_feature[:, 1, :]
        else:
            raise ValueError(
                f"vec_feature must be (B, 6) or (B, 2, 3), got {tuple(vec_feature.shape)}"
            )
        return dir1, dir2

    def _build_input(self, mix_feat: Tensor, emb: Optional[Tensor]) -> Tensor:
        if self.cond_dim == 0:
            return mix_feat
        if emb is None:
            raise ValueError("cond_dim > 0 but emb is None")
        B, Freq, Time, _ = mix_feat.shape
        dir_map = emb[:, None, None, :].expand(B, Freq, Time, self.cond_dim)
        return torch.cat([mix_feat, dir_map], dim=-1)

    def forward(self, audio_mix: Tensor, vec_feature: Tensor, return_masks: bool = False):
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

        dir1, dir2 = self._split_vec(vec_feature)
        emb1 = self.DirVecNet(dir1) if self.DirVecNet is not None else None
        emb2 = self.DirVecNet(dir2) if self.DirVecNet is not None else None

        x1 = self._build_input(mix_feat, emb1)
        x2 = self._build_input(mix_feat, emb2)

        pred1 = self.mask_net(x1)  # (B, F, T, output_nc)
        pred2 = self.mask_net(x2)

        pred1 = pred1.permute(0, 3, 1, 2)
        pred2 = pred2.permute(0, 3, 1, 2)
        predictions = torch.stack([pred1, pred2], dim=1)  # (B, 2, output_nc, F, T)

        masks = self._apply_activation(predictions)

        sep_wavs = []
        for src_idx in range(2):
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
        if return_masks:
            return sep_audio, masks
        return sep_audio


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=7, python -m models.arch.SpatialNet
    x = torch.randn((1, 129, 251, 12))  #.cuda() # 251 = 4 second; 129 = 8 kHz; 257 = 16 kHz
    spatialnet_small = SpatialNet(
        dim_input=12,
        dim_output=4,
        num_layers=8,
        dim_hidden=96,
        dim_ffn=192,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=8,
        num_freqs=129,
        full_share=0,
    )  #.cuda()
    # from packaging.version import Version
    # if Version(torch.__version__) >= Version('2.0.0'):
    #     SSFNet_small = torch.compile(SSFNet_small)
    # torch.cuda.synchronize(7)
    import time
    ts = time.time()
    y = spatialnet_small(x)
    # torch.cuda.synchronize(7)
    te = time.time()
    print(spatialnet_small)
    print(y.shape)
    print(te - ts)

    spatialnet_small = spatialnet_small.to('meta')
    x = x.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(spatialnet_small, display=False) as fcm:
        y = spatialnet_small(x)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in spatialnet_small.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")
