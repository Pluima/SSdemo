import math
from typing import Optional, Tuple

import torch


def normalize_audio_loudness(
    audio: torch.Tensor, target_lufs: float = -23.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize audio loudness to a target LUFS-like level using RMS.

    Returns:
        normalized_audio, gain_linear
    """
    if audio.numel() == 0:
        return audio, torch.tensor(1.0, device=audio.device, dtype=audio.dtype)

    rms = torch.sqrt(torch.mean(audio ** 2))
    if rms == 0:
        return audio, torch.tensor(1.0, device=audio.device, dtype=audio.dtype)

    current_lufs = 20.0 * torch.log10(rms)
    gain_linear = 10.0 ** ((target_lufs - current_lufs) / 20.0)
    normalized = audio * gain_linear

    max_val = torch.max(torch.abs(normalized))
    if max_val > 1.0:
        clipping_gain = 0.95 / max_val
        normalized = normalized * clipping_gain
        gain_linear = gain_linear * clipping_gain

    return normalized, gain_linear


class TFNetStreamer:
    """
    Streaming wrapper for TFNetSeparator using a ring buffer and middle cropping.

    Strategy:
      - Maintain a waveform ring buffer: [left_context | emit_len | right_context]
      - Append each new chunk (emit_len), shift the buffer
      - Run TFNetSeparator on the full buffer
      - Return only the center segment to avoid boundary artifacts
    """

    def __init__(
        self,
        model: torch.nn.Module,
        sr: int,
        emit_ms: float = 40.0,
        left_ms: float = 200.0,
        right_ms: float = 200.0,
        device: str = "cuda",
        input_channels: int = 1,
        normalize_input: bool = True,
        target_lufs: float = -23.0,
        warmup_steps: Optional[int] = None,
        return_stereo: bool = False,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.sr = sr
        self.input_channels = input_channels
        self.normalize_input = normalize_input
        self.target_lufs = target_lufs
        self.return_stereo = return_stereo

        self.emit_len = int(sr * emit_ms / 1000)
        self.left_len = int(sr * left_ms / 1000)
        self.right_len = int(sr * right_ms / 1000)
        self.win_len = self.left_len + self.emit_len + self.right_len

        if self.emit_len <= 0:
            raise ValueError(f"emit_len must be > 0, got {self.emit_len}")
        if self.win_len <= self.emit_len:
            raise ValueError("win_len must be larger than emit_len")

        buf_shape = (self.input_channels, self.win_len) if self.input_channels > 1 else (self.win_len,)
        self.buf = torch.zeros(buf_shape, device=self.device)

        if warmup_steps is None:
            warmup_steps = max(1, math.ceil(self.right_len / self.emit_len))
        self.warmup_steps = warmup_steps
        self.warmup_count = 0

    def reset(self) -> None:
        self.buf.zero_()
        self.warmup_count = 0

    def _shift_and_append(self, x_chunk: torch.Tensor) -> None:
        if self.input_channels > 1:
            self.buf[:, :-self.emit_len] = self.buf[:, self.emit_len:]
            self.buf[:, -self.emit_len:] = x_chunk
        else:
            self.buf[:-self.emit_len] = self.buf[self.emit_len:]
            self.buf[-self.emit_len:] = x_chunk

    @torch.inference_mode()
    def process(self, x_chunk, vec_feature):
        """
        Args:
            x_chunk: (emit_len,) or (C, emit_len) waveform chunk
            vec_feature: (6,) or (2,3) direction vectors
        Returns:
            y_mid: (2, emit_len) or (2, 2, emit_len) if return_stereo=True
        """
        x_chunk = torch.as_tensor(x_chunk, device=self.device).float()
        if self.input_channels > 1:
            if x_chunk.dim() == 1:
                x_chunk = x_chunk.unsqueeze(0).repeat(self.input_channels, 1)
            if x_chunk.shape != (self.input_channels, self.emit_len):
                raise ValueError(
                    f"x_chunk must be ({self.input_channels}, {self.emit_len}), got {tuple(x_chunk.shape)}"
                )
        else:
            if x_chunk.dim() != 1 or x_chunk.numel() != self.emit_len:
                raise ValueError(f"x_chunk must be ({self.emit_len},), got {tuple(x_chunk.shape)}")

        self._shift_and_append(x_chunk)

        if self.warmup_count < self.warmup_steps:
            self.warmup_count += 1
            if self.return_stereo:
                return torch.zeros((2, 2, self.emit_len), device=self.device)
            return torch.zeros((2, self.emit_len), device=self.device)

        if self.input_channels > 1:
            buf = self.buf.unsqueeze(0)  # (1, C, T)
        else:
            buf = self.buf.unsqueeze(0)  # (1, T)

        gain = None
        if self.normalize_input:
            buf, gain = normalize_audio_loudness(buf, target_lufs=self.target_lufs)

        v = torch.as_tensor(vec_feature, device=self.device).float().unsqueeze(0)
        y = self.model(buf, v)[0]  # (2, T) or (2, 2, T)

        y_mid = y[..., self.left_len : self.left_len + self.emit_len]

        if not self.return_stereo and y_mid.dim() == 3:
            y_mid = y_mid.mean(dim=1)

        if gain is not None:
            y_mid = y_mid / gain

        return y_mid
