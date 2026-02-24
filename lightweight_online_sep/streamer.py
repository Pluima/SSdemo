from typing import Any, Optional

import numpy as np
import torch


class OnlineSeparatorStreamer:
    """Stateful low-latency streaming wrapper for LightweightCausalSeparator."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)

        self.n_fft = int(model.n_fft)
        self.hop_length = int(model.hop_length)
        self.win_length = int(model.win_length)
        self.num_speakers = int(model.num_speakers)

        self.window = model.analysis_window.to(self.device)
        self.window_sq = (self.window ** 2)

        self._eps = 1e-8
        self.reset()

    def reset(self):
        self.hidden: Optional[Any] = None
        self.analysis_buffer = torch.zeros(self.win_length - self.hop_length, device=self.device)
        self.ola_num = torch.zeros(self.num_speakers, self.win_length, device=self.device)
        self.ola_den = torch.zeros(self.win_length, device=self.device)
        self.output_buffer = torch.zeros(self.num_speakers, 0, device=self.device)

    @torch.no_grad()
    def _emit_ready_frames(self) -> torch.Tensor:
        emitted = []

        while self.analysis_buffer.numel() >= self.win_length:
            frame = self.analysis_buffer[: self.win_length]
            self.analysis_buffer = self.analysis_buffer[self.hop_length :]

            frame_win = frame * self.window
            mix_spec = torch.fft.rfft(frame_win, n=self.n_fft).unsqueeze(0)  # [1, F]
            est_spec, self.hidden, _ = self.model.forward_step(
                mix_spec,
                self.hidden,
                frame_time=frame.unsqueeze(0),
            )  # [1, S, F]

            est_frame = torch.fft.irfft(est_spec.squeeze(0), n=self.n_fft)[:, : self.win_length]
            est_frame = est_frame * self.window.unsqueeze(0)

            self.ola_num = self.ola_num + est_frame
            self.ola_den = self.ola_den + self.window_sq

            out_num = self.ola_num[:, : self.hop_length]
            out_den = torch.clamp(self.ola_den[: self.hop_length], min=self._eps).unsqueeze(0)
            emitted.append(out_num / out_den)

            self.ola_num = torch.cat(
                [
                    self.ola_num[:, self.hop_length :],
                    torch.zeros(self.num_speakers, self.hop_length, device=self.device),
                ],
                dim=1,
            )
            self.ola_den = torch.cat(
                [
                    self.ola_den[self.hop_length :],
                    torch.zeros(self.hop_length, device=self.device),
                ],
                dim=0,
            )

        if not emitted:
            return torch.zeros(self.num_speakers, 0, device=self.device)
        return torch.cat(emitted, dim=1)

    @torch.no_grad()
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Args:
            chunk: mono audio chunk with shape [T]
        Returns:
            separated chunk [S, T]
        """
        chunk_tensor = torch.as_tensor(chunk, dtype=torch.float32, device=self.device).flatten()
        in_len = int(chunk_tensor.numel())

        self.analysis_buffer = torch.cat([self.analysis_buffer, chunk_tensor], dim=0)
        new_out = self._emit_ready_frames()
        if new_out.numel() > 0:
            self.output_buffer = torch.cat([self.output_buffer, new_out], dim=1)

        if self.output_buffer.shape[1] < in_len:
            pad_len = in_len - self.output_buffer.shape[1]
            pad = torch.zeros(self.num_speakers, pad_len, device=self.device)
            self.output_buffer = torch.cat([self.output_buffer, pad], dim=1)

        out = self.output_buffer[:, :in_len]
        self.output_buffer = self.output_buffer[:, in_len:]
        return out.detach().cpu().numpy()

    @torch.no_grad()
    def flush(self) -> np.ndarray:
        """Flush residual states by feeding zero paddings once."""
        tail = np.zeros(self.win_length, dtype=np.float32)
        _ = self.process_chunk(tail)
        if self.output_buffer.shape[1] == 0:
            return np.zeros((self.num_speakers, 0), dtype=np.float32)
        out = self.output_buffer.detach().cpu().numpy()
        self.output_buffer = torch.zeros(self.num_speakers, 0, device=self.device)
        return out
