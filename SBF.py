import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DSBF(nn.Module):
    """
    Dual-Source Spatial Beamforming (DSBF) for stereo speech enhancement
    with spatial-cue preservation based on the paper:
    "Real-time Stereo Speech Enhancement with Spatial-Cue Preservation
    based on Dual-Path Structure" by Togami et al. (2024)

    This module performs spatial beamforming to separate two speech sources
    while preserving spatial cues using a dual-path structure.
    """

    def __init__(self, n_fft=512, hop_length=128, num_sources=2, n_channels=2,
                 adaptive_steering=True, alpha=0.9):
        """
        Args:
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            num_sources: Number of sources to separate (default: 2)
            n_channels: Number of input channels (default: 2 for stereo)
            adaptive_steering: Whether to adaptively update steering vectors
            alpha: Smoothing factor for adaptive steering vector update
        """
        super(DSBF, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_sources = num_sources
        self.n_channels = n_channels
        self.adaptive_steering = adaptive_steering
        self.alpha = alpha

        # Frequency bins
        self.n_freq = n_fft // 2 + 1

        # Initialize steering vectors for each source
        # Shape: [num_sources, n_freq, n_channels]
        self.register_buffer('steering_vectors',
                           torch.randn(num_sources, self.n_freq, n_channels, dtype=torch.complex64))

        # Initialize steering vectors to reasonable values (unit vectors)
        for src in range(num_sources):
            for f in range(self.n_freq):
                # Random initialization with unit norm
                vec = torch.randn(n_channels, dtype=torch.complex64)
                vec = vec / torch.norm(vec)
                self.steering_vectors[src, f] = vec

    def stft(self, x):
        """
        Apply STFT to input signal
        Args:
            x: [batch, channels, time]
        Returns:
            X: [batch, channels, freq, time]
        """
        batch_size, n_channels, time_len = x.shape

        # Apply STFT to each channel
        X_list = []
        for ch in range(n_channels):
            X_ch = torch.stft(x[:, ch], n_fft=self.n_fft, hop_length=self.hop_length,
                            window=torch.hann_window(self.n_fft).to(x.device),
                            return_complex=True)
            X_list.append(X_ch)

        # Stack channels: [batch, freq, time, channels] -> [batch, channels, freq, time]
        X = torch.stack(X_list, dim=1)
        return X

    def istft(self, X):
        """
        Apply inverse STFT
        Args:
            X: [batch, channels, freq, time]
        Returns:
            x: [batch, channels, time]
        """
        batch_size, n_channels, n_freq, n_frames = X.shape

        x_list = []
        for ch in range(n_channels):
            x_ch = torch.istft(X[:, ch], n_fft=self.n_fft, hop_length=self.hop_length,
                             window=torch.hann_window(self.n_fft).to(X.device),
                             return_complex=False)
            x_list.append(x_ch)

        # Stack channels back
        x = torch.stack(x_list, dim=1)
        return x

    def compute_mvdr_weights(self, X, source_idx):
        """
        Compute MVDR beamformer weights for a specific source
        Args:
            X: [batch, channels, freq, time] - input STFT
            source_idx: Index of the source to enhance
        Returns:
            W: [batch, channels, freq, time] - beamformer weights
        """
        batch_size, n_channels, n_freq, n_frames = X.shape

        # Get steering vector for this source: [n_freq, n_channels]
        steering_vec = self.steering_vectors[source_idx]  # [n_freq, n_channels]

        # Compute spatial covariance matrix for noise (assuming diffuse noise)
        # For simplicity, use identity matrix as noise covariance
        # In practice, this should be estimated from noise-only segments
        noise_cov = torch.eye(n_channels, dtype=X.dtype, device=X.device)
        noise_cov = noise_cov.unsqueeze(0).unsqueeze(0)  # [1, 1, channels, channels]

        # Compute MVDR weights for each frequency and time frame
        W = torch.zeros_like(X)

        for b in range(batch_size):
            for f in range(n_freq):
                for t in range(n_frames):
                    # Steering vector for this frequency: [channels]
                    a_f = steering_vec[f]

                    # Input signal for this frequency and time: [channels]
                    x_ft = X[b, :, f, t]

                    # Compute MVDR weight: w = (Φ_nn^-1 @ a) / (a^H @ Φ_nn^-1 @ a)
                    # Full MVDR would require noise covariance matrix Φ_nn
                    # Simplified version assuming Φ_nn = I (identity matrix)
                    # This assumes isotropic noise and simplifies to delay-and-sum beamforming
                    w = a_f / torch.sum(torch.abs(a_f)**2)
                    W[b, :, f, t] = w

        return W

    def apply_beamforming(self, X, source_idx):
        """
        Apply beamforming to extract one source
        Args:
            X: [batch, channels, freq, time] - input STFT
            source_idx: Index of the source to extract
        Returns:
            Y: [batch, freq, time] - extracted source in STFT domain
        """
        # Compute MVDR weights
        W = self.compute_mvdr_weights(X, source_idx)

        # Apply beamforming: Y = w^H @ X
        # W: [batch, channels, freq, time], X: [batch, channels, freq, time]
        Y = torch.sum(W.conj() * X, dim=1)  # [batch, freq, time]

        return Y

    def update_steering_vectors(self, Y_enhanced, X_input):
        """
        Adaptively update steering vectors using enhanced output
        Args:
            Y_enhanced: [batch, freq, time] - enhanced source STFT
            X_input: [batch, channels, freq, time] - input mixture STFT
        """
        if not self.adaptive_steering:
            return

        batch_size, n_channels, n_freq, n_frames = X_input.shape

        for src in range(self.num_sources):
            # Compute correlation between enhanced signal and input channels
            for f in range(n_freq):
                # Enhanced signal for this frequency: [batch, time]
                y_f = Y_enhanced[:, f, :]

                # Input channels for this frequency: [batch, channels, time]
                x_f = X_input[:, :, f, :]

                # Compute cross-correlation: [batch, channels]
                corr = torch.mean(y_f.unsqueeze(1).conj() * x_f, dim=-1)

                # Update steering vector with smoothing
                new_steering = corr / (torch.norm(corr, dim=-1, keepdim=True) + 1e-8)
                new_steering = torch.mean(new_steering, dim=0)  # Average across batch

                # Smooth update
                self.steering_vectors[src, f] = (self.alpha * self.steering_vectors[src, f] +
                                               (1 - self.alpha) * new_steering)

    def apply_common_gain(self, Y_mono, target_channels=2):
        """
        Apply common gain across channels to preserve spatial cues
        Args:
            Y_mono: [batch, freq, time] - monaural enhanced signal
            target_channels: Number of output channels (default: 2 for stereo)
        Returns:
            Y_stereo: [batch, channels, freq, time] - stereo enhanced signal
        """
        batch_size, n_freq, n_frames = Y_mono.shape

        # For simplicity, replicate the monaural signal to all channels
        # In practice, this should apply the same gain to maintain spatial cues
        Y_stereo = Y_mono.unsqueeze(1).repeat(1, target_channels, 1, 1)

        return Y_stereo

    def forward(self, x, enhanced_sources=None):
        """
        Forward pass of DSBF
        Args:
            x: [batch, channels, time] - input stereo mixture
            enhanced_sources: Optional [batch, num_sources, freq, time] - pre-enhanced sources
        Returns:
            y_separated: [batch, num_sources, channels, time] - separated stereo sources
            y_beamformed: [batch, num_sources, time] - beamformed monaural sources
        """
        # Convert to STFT domain
        X = self.stft(x)  # [batch, channels, freq, time]

        batch_size, n_channels, n_freq, n_frames = X.shape

        # Separate each source using beamforming
        y_beamformed = []
        y_separated = []

        for src in range(self.num_sources):
            # Apply beamforming to get monaural source
            Y_mono = self.apply_beamforming(X, src)  # [batch, freq, time]

            # Apply common gain to create stereo output
            Y_stereo = self.apply_common_gain(Y_mono, target_channels=n_channels)

            y_beamformed.append(Y_mono)
            y_separated.append(Y_stereo)

            # Update steering vectors if enhanced sources are provided
            if enhanced_sources is not None and src < enhanced_sources.shape[1]:
                Y_enhanced = enhanced_sources[:, src]  # [batch, freq, time]
                self.update_steering_vectors(Y_enhanced, X)

        # Stack results
        y_beamformed = torch.stack(y_beamformed, dim=1)  # [batch, num_sources, freq, time]
        y_separated = torch.stack(y_separated, dim=1)     # [batch, num_sources, channels, freq, time]

        # Convert back to time domain
        y_time_domain = []
        for src in range(self.num_sources):
            y_src = self.istft(y_separated[:, src])  # [batch, channels, time]
            y_time_domain.append(y_src)

        y_time_domain = torch.stack(y_time_domain, dim=1)  # [batch, num_sources, channels, time]

        return y_time_domain, y_beamformed


def test_dsbf():
    """Test function for DSBF module"""
    # Create test input
    batch_size = 2
    n_channels = 2
    time_len = 16000  # 1 second at 16kHz

    # Create dummy stereo mixture
    x = torch.randn(batch_size, n_channels, time_len)

    # Initialize DSBF
    dsbf = DSBF(n_fft=512, hop_length=128)

    # Forward pass
    y_separated, y_beamformed = dsbf(x)

    print(f"Input shape: {x.shape}")
    print(f"Separated output shape: {y_separated.shape}")
    print(f"Beamformed output shape: {y_beamformed.shape}")

    # Test with enhanced sources for adaptive update
    enhanced_sources = torch.randn(batch_size, 2, dsbf.n_freq, y_beamformed.shape[-1])
    y_separated_adapt, y_beamformed_adapt = dsbf(x, enhanced_sources)

    print(f"Adaptive separated output shape: {y_separated_adapt.shape}")
    print("DSBF test completed successfully!")


if __name__ == "__main__":
    test_dsbf()
