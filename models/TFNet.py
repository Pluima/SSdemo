
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import copy
import math
import numpy as np
from torch.autograd import Variable
"""
This file contains:
- Torch-native STFT/iSTFT helpers (no librosa dependency)
- A conditional 2-source U-Net that predicts complex masks
- A wrapper model (TFNetSeparator) whose forward returns separated waveforms directly
"""


def _twoch_to_complex(two_ch):
    """(B, 2, F, T) -> (B, F, T) complex"""
    real = two_ch[:, 0]
    imag = two_ch[:, 1]
    # torch.complex does not accept bfloat16
    if real.dtype == torch.bfloat16:
        real = real.float()
        imag = imag.float()
    return torch.complex(real, imag)


def _complex_to_twoch(cpx):
    """(B, F, T) complex -> (B, 2, F, T)"""
    return torch.stack((cpx.real, cpx.imag), dim=1)


def _complex_to_twoch_multi(cpx):
    """(B, C, F, T) complex -> (B, 2*C, F, T)"""
    return torch.cat((cpx.real, cpx.imag), dim=1)



def _group_norm(channels, num_groups=8):
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def unet_conv(input_nc, output_nc, norm_layer=_group_norm):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=_group_norm, kernel_size=4):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv])


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        attn = self.mlp(avg_pool) + self.mlp(max_pool)
        return x * self.sigmoid(attn)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class LiteCBAM(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, norm_layer=_group_norm):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            norm_layer(ch_out),
            nn.LeakyReLU(0.2, True)
        )
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, outermost=False, norm_layer=_group_norm):
        super(up_conv,self).__init__()
        if not outermost:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2.,1.)),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                norm_layer(ch_out),
                nn.ReLU(inplace=True)
                )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=(2.,1.)),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.Sigmoid()
                )
    def forward(self,x):
        x = self.up(x)
        return x


class Unet(nn.Module):
    """
    Conditional U-Net for (complex) mask prediction on spectrograms.

    This version supports *two* sources controlled by two 3D direction vectors:
      - vec_feature: shape (B, 6) where [:, :3] is source-1 direction and [:, 3:] is source-2 direction
        or shape (B, 2, 3).

    Output:
      - stacked masks of shape (B, 2, output_nc, F, T) where dim=1 indexes sources.
        By default output_nc=2 corresponds to (real, imag) complex mask channels.
        For stereo masks per source, set output_nc=4 for (L_real, L_imag, R_real, R_imag).
    """
    def __init__(
        self,
        ngf=64,
        input_nc=2,
        output_nc=2,
        cond_dim=128,
        vec_dim_per_source=3,
        use_vector_cue=True,
        bottleneck_blocks=2,
        bottleneck_residual=True,
        bottleneck_attention=False,
        attention_reduction=8,
        attention_kernel=7,
    ):
        super(Unet, self).__init__()
        self.cond_dim = cond_dim
        self.use_vector_cue = bool(use_vector_cue)
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = conv_block(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = conv_block(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = conv_block(ngf * 8, ngf * 8)
        self.audionet_convlayer8 = conv_block(ngf * 8, ngf * 8)
        self.frequency_pool = nn.MaxPool2d([2,1])

        # Optional extra capacity at the bottleneck (keeps spatial sizes).
        self.bottleneck_residual = bottleneck_residual
        self.bottleneck_blocks = nn.ModuleList(
            [conv_block(ngf * 8, ngf * 8) for _ in range(max(0, bottleneck_blocks))]
        )
        self.bottleneck_attention = bottleneck_attention
        self.attention = LiteCBAM(
            ngf * 8,
            reduction=attention_reduction,
            spatial_kernel=attention_kernel,
        ) if bottleneck_attention else None
        
        # Per-source direction embedding: (B, 3) -> (B, cond_dim)
        # Use LayerNorm to avoid GroupNorm issues with small batch sizes.
        self.DirVecNet = nn.Sequential(
            nn.Linear(vec_dim_per_source, cond_dim),
            # nn.Dropout(0.1),
            # nn.LayerNorm(cond_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(cond_dim, cond_dim),
            # nn.Dropout(0.1),
            # nn.LayerNorm(cond_dim),
            # nn.LeakyReLU(0.2, True),
        )
        if not self.use_vector_cue:
            for p in self.DirVecNet.parameters():
                p.requires_grad = False
        
        # Decoder processes one source at a time.
        # First decoder block consumes [emb, bottleneck_audio] concatenated on channel dim
        self.audionet_upconvlayer1 = up_conv(cond_dim + ngf * 8, ngf * 8)
        self.audionet_upconvlayer2 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = up_conv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer5 = up_conv(ngf * 16, ngf *4)
        self.audionet_upconvlayer6 = up_conv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 4, ngf)
        # Output layer produces masks for one source: output_nc channels
        self.audionet_upconvlayer8 = unet_upconv(ngf * 2 + cond_dim, output_nc, True)
        self.Sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()

    def _encode(self, audio_mix_spec):
        # Expect (B, C, F, T). Many STFT pipelines produce F=n_fft//2+1=1025; crop to 1024 for clean downsampling.
        audio_conv1feature = self.audionet_convlayer1(audio_mix_spec[:, :, :-1, :])
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)

        audio_conv3feature = self.frequency_pool(self.audionet_convlayer3(audio_conv2feature))
        audio_conv4feature = self.frequency_pool(self.audionet_convlayer4(audio_conv3feature))
        audio_conv5feature = self.frequency_pool(self.audionet_convlayer5(audio_conv4feature))
        audio_conv6feature = self.frequency_pool(self.audionet_convlayer6(audio_conv5feature))
        audio_conv7feature = self.frequency_pool(self.audionet_convlayer7(audio_conv6feature))
        audio_conv8feature = self.frequency_pool(self.audionet_convlayer8(audio_conv7feature))
        for block in self.bottleneck_blocks:
            if self.bottleneck_residual:
                audio_conv8feature = audio_conv8feature + block(audio_conv8feature)
            else:
                audio_conv8feature = block(audio_conv8feature)
        if self.attention is not None:
            audio_conv8feature = self.attention(audio_conv8feature)

        return (
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
        )

    def _decode_single(self, audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature,
                       audio_conv5feature, audio_conv6feature, audio_conv7feature, audio_conv8feature,
                       emb):
        """
        Decode one source.
        emb: (B, cond_dim)
        Returns: (B, output_nc, F, T)
        """
        # Concat direction embedding at bottleneck
        dir_map = emb.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, audio_conv8feature.size(2), audio_conv8feature.size(3)
        )

        x = torch.cat((dir_map, audio_conv8feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(x)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv7feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv6feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv5feature), dim=1))
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv4feature), dim=1))
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv3feature), dim=1))
        audio_upconv7feature = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv2feature), dim=1))

        # Dynamic alignment for odd input lengths
        if audio_upconv7feature.shape[2:] != audio_conv1feature.shape[2:]:
            audio_upconv7feature = F.interpolate(
                audio_upconv7feature,
                size=audio_conv1feature.shape[2:],
                mode='bilinear',
                align_corners=False,
            )
        
        # Concat direction embedding again before final layer
        dir_map = emb.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, audio_upconv7feature.size(2), audio_upconv7feature.size(3)
        )

        audio_upconv7feature = torch.cat((audio_upconv7feature, dir_map), dim=1)
        
        # Output: (B, output_nc, F, T)
        prediction = self.audionet_upconvlayer8(torch.cat((audio_upconv7feature, audio_conv1feature), dim=1))
        return prediction

    def forward(self, audio_mix_spec, vec_feature=None, activation='Sigmoid', return_tuple=False):
        """
        audio_mix_spec: (B, 2, F, T)
        vec_feature: (B, 6) or (B, 2, 3)

        Returns:
          - if return_tuple=False: (B, 2, output_nc, F, T)
          - if return_tuple=True: (mask1, mask2) each (B, output_nc, F, T)
        """
        feats = self._encode(audio_mix_spec)
        (
            audio_conv1feature,
            audio_conv2feature,
            audio_conv3feature,
            audio_conv4feature,
            audio_conv5feature,
            audio_conv6feature,
            audio_conv7feature,
            audio_conv8feature,
        ) = feats

        if (not self.use_vector_cue) or (vec_feature is None):
            # Direction branch disabled: use zero conditioning.
            emb1 = torch.zeros(
                audio_mix_spec.size(0),
                self.cond_dim,
                device=audio_mix_spec.device,
                dtype=audio_mix_spec.dtype,
            )
            emb2 = torch.zeros(
                audio_mix_spec.size(0),
                self.cond_dim,
                device=audio_mix_spec.device,
                dtype=audio_mix_spec.dtype,
            )
        elif vec_feature.dim() == 2 and vec_feature.size(1) == 6:
            dir1 = vec_feature[:, :3]
            dir2 = vec_feature[:, 3:]
            emb1 = self.DirVecNet(dir1)
            emb2 = self.DirVecNet(dir2)
        elif vec_feature.dim() == 3 and vec_feature.size(1) == 2 and vec_feature.size(2) == 3:
            dir1 = vec_feature[:, 0, :]
            dir2 = vec_feature[:, 1, :]
            emb1 = self.DirVecNet(dir1)
            emb2 = self.DirVecNet(dir2)
        else:
            raise ValueError(f"vec_feature must be (B, 6) or (B, 2, 3), got {tuple(vec_feature.shape)}")

        # Decode each source separately, then merge outputs
        pred1 = self._decode_single(
            audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature,
            audio_conv5feature, audio_conv6feature, audio_conv7feature, audio_conv8feature,
            emb1
        )
        pred2 = self._decode_single(
            audio_conv1feature, audio_conv2feature, audio_conv3feature, audio_conv4feature,
            audio_conv5feature, audio_conv6feature, audio_conv7feature, audio_conv8feature,
            emb2
        )

        predictions = torch.stack([pred1, pred2], dim=1)
        # predictions: (B, 2, output_nc, F, T)

        if activation == 'Sigmoid':
            masks = self.Sigmoid(predictions)
        elif activation == 'Tanh':
            masks = 4 * self.Tanh(predictions)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if return_tuple:
            mask1 = masks[:, 0]  # (B, output_nc, F, T)
            mask2 = masks[:, 1]  # (B, output_nc, F, T)
            return mask1, mask2
        return masks  # (B, 2, output_nc, F, T)

def apply_complex_mask(spectrogram, mask):
    """
    Apply complex mask to spectrogram.
    spectrogram: (B, 2, F, T)
    mask: (B, 2, F, T)
    Returns: (B, 2, F, T)
    """
    s_real = spectrogram[:, 0]
    s_imag = spectrogram[:, 1]
    m_real = mask[:, 0]
    m_imag = mask[:, 1]
    
    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    out_real = s_real * m_real - s_imag * m_imag
    out_imag = s_real * m_imag + s_imag * m_real
    
    return torch.stack((out_real, out_imag), dim=1)




class TFNetSeparator(nn.Module):
    """
    End-to-end separator:
      audio_mix (B, T) or (B, 2, T) + vec_feature (B, 6) -> separated waveforms (B, 2, T)

    Internals:
      STFT -> (complex) -> 2ch/4ch -> Unet -> 2 complex masks -> apply -> iSTFT
    """
    def __init__(self, args):
        super().__init__()
        self.stft_frame = args.network_audio['stft_frame']
        self.stft_hop = args.network_audio['stft_hop']
        self.n_fft = args.network_audio['n_fft']
        self.activation = args.network_audio['activation']
        self.stereo_loss = args.stereo_loss
        # Window as buffer so it moves with .to(device)
        self.register_buffer("window", torch.hann_window(self.stft_frame), persistent=False)

        input_nc = args.network_audio['input_nc']
        if hasattr(args, "input_mono") and not args.input_mono:
            if input_nc == 2:
                input_nc = 4
        self.input_nc = input_nc
        output_nc = args.network_audio['output_nc']

        # Mask network: outputs (B, 2, output_nc, Fmask, Tmask)
        self.mask_net = Unet(
            ngf=args.network_audio['ngf'],
            input_nc=input_nc,
            output_nc=output_nc,
            cond_dim=args.network_audio['cond_dim'],
            use_vector_cue=bool(getattr(args, "vector_cue", 1)),
            bottleneck_blocks=args.network_audio.get('bottleneck_blocks', 2),
            bottleneck_residual=args.network_audio.get('bottleneck_residual', True),
            bottleneck_attention=args.network_audio.get('bottleneck_attention', False),
            attention_reduction=args.network_audio.get('attention_reduction', 8),
            attention_kernel=args.network_audio.get('attention_kernel', 7),
        )

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

    def forward(self, audio_mix, vec_feature=None, return_masks=False):
        """
        Args:
            audio_mix: (B, T), (B, 2, T), or (T,)
            vec_feature: (B, 6) or (B, 2, 3)

        Returns:
            sep_audio: (B, 2, T) or (B, 2, 2, T) for stereo output
            (optional) masks: (B, 2, output_nc, Fmask_aligned, Tmask_aligned)
        """
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

        # 1) STFT (complex)
        if stereo_input:
            mix_cpx = torch.stack(
                [self._stft(audio_mix[:, ch, :]) for ch in range(2)],
                dim=1,
            )  # (B, 2, F, Tspec)
        else:
            mix_cpx = self._stft(audio_mix)  # (B, F, Tspec)

        # 2) To 2-channel (mono) or 4-channel (stereo) for Unet input
        if stereo_input:
            mix_twoch = _complex_to_twoch_multi(mix_cpx)  # (B, 4, F, Tspec)
        else:
            mix_twoch = _complex_to_twoch(mix_cpx)  # (B, 2, F, Tspec)

        # 3) Predict 2 masks (complex as 2-channel per stream)
        masks = self.mask_net(mix_twoch, vec_feature, activation=self.activation, return_tuple=False)
        # masks: (B, 2, output_nc, Fmask, Tmask)

        # 4) Apply masks on cropped spectrum (match Unet internal Nyquist drop)
        if stereo_input:
            mix_cpx_crop = mix_cpx[:, :, :-1, :]  # (B, 2, F-1, Tspec)
            target_F, target_T = mix_cpx_crop.shape[2], mix_cpx_crop.shape[3]
        else:
            mix_twoch_crop = mix_twoch[:, :, :-1, :]        # (B, 2, F-1, Tspec)
            mix_cpx_crop = _twoch_to_complex(mix_twoch_crop)  # (B, F-1, Tspec) complex
            target_F, target_T = mix_twoch_crop.shape[2], mix_twoch_crop.shape[3]
        if (masks.shape[3], masks.shape[4]) != (target_F, target_T):
            # (B, S, C, F, T) -> (B*S, C, F, T) -> interpolate -> reshape back
            Bb, S, Cc, Fm, Tm = masks.shape
            masks_4d = masks.reshape(Bb * S, Cc, Fm, Tm)
            masks_4d = F.interpolate(
                masks_4d,
                size=(target_F, target_T),
                mode="bilinear",
                align_corners=False,
            )
            masks = masks_4d.reshape(Bb, S, Cc, target_F, target_T)

        src_cpx_crops = []
        # sep_wavs = []
        for src_idx in range(2):
            mask_src = masks[:, src_idx]  # (B, output_nc, F-1, Tspec)
            if stereo_input:
                if mask_src.size(1) != 4:
                    raise ValueError(
                        f"stereo mask requires output_nc=4 (L_real,L_imag,R_real,R_imag), got {mask_src.size(1)}"
                    )
                mask_L = mask_src[:, 0:2]  # (B, 2, F-1, Tspec)
                mask_R = mask_src[:, 2:4]  # (B, 2, F-1, Tspec)
                mask_L_cpx = _twoch_to_complex(mask_L)  # (B, F-1, Tspec) complex
                mask_R_cpx = _twoch_to_complex(mask_R)  # (B, F-1, Tspec) complex
                src_L = mix_cpx_crop[:, 0] * mask_L_cpx  # (B, F-1, Tspec)
                src_R = mix_cpx_crop[:, 1] * mask_R_cpx  # (B, F-1, Tspec)
                src_cpx_crop = torch.stack([src_L, src_R], dim=1)  # (B, 2, F-1, Tspec)
            else:
                mask_twoch = mask_src  # (B, 2, F-1, Tspec)
                mask_cpx = _twoch_to_complex(mask_twoch)  # (B, F-1, Tspec) complex
                src_cpx_crop = mix_cpx_crop * mask_cpx  # (B, F-1, Tspec) complex
            src_cpx_crops.append(src_cpx_crop)

        # Stack sources without mixture consistency projection
        src_cpx_crops = torch.stack(src_cpx_crops, dim=1)

        sep_wavs = []
        for src_idx in range(2):
            src_cpx_crop = src_cpx_crops[:, src_idx]
            if stereo_input:
                # Pad Nyquist bin back for iSTFT (zeros)
                src_cpx = F.pad(src_cpx_crop, (0, 0, 0, 1))  # (B, 2, F, Tspec)
                src_wavs = []
                for ch in range(src_cpx.size(1)):
                    src_wavs.append(self._istft(src_cpx[:, ch], length=Tlen))  # (B, T)
                if self.stereo_loss:
                    src_wav = torch.stack(src_wavs, dim=1)  # (B, 2, T)
                else:
                    src_wav = torch.stack(src_wavs, dim=1).mean(dim=1)  # (B, 2, T)
            else:
                # Pad Nyquist bin back for iSTFT (zeros)
                src_cpx = F.pad(src_cpx_crop, (0, 0, 0, 1))  # pad freq dim: (B, F, Tspec)
                src_wav = self._istft(src_cpx, length=Tlen)  # (B, T)
            sep_wavs.append(src_wav)

        sep_audio = torch.stack(sep_wavs, dim=1)  # (B, 2, T)

        if return_masks:
            return sep_audio, masks
        return sep_audio

if __name__ == "__main__":
    # Quick sanity test for the wrapper model
    model = TFNetSeparator()
    model.eval()

    x = torch.randn(2, 32000)  # (B, T)
    v = torch.randn(2, 6)      # (B, 6)
    y = model(x, v)
    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))  # (B, 2, T)
