import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from itertools import permutations
import itertools
EPS = 1e-6


def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))


def Loss( egs,ests):
    # spks x n x S
    refs = egs
    num_spks = len(refs)

    def sisnr_loss(permute):
        # for one permute
        return sum(
            [sisnr(ests[s], refs[t])
             for s, t in enumerate(permute)]) / len(permute)
             # average the value

    # P x N
    N = egs[0].size(0)
    sisnr_mat = torch.stack(
        [sisnr_loss(p) for p in permutations(range(num_spks))])
    max_perutt, _ = torch.max(sisnr_mat, dim=0)
    # si-snr
    return -torch.sum(max_perutt) / N

def loss_moss(labels,preds):
    pit_si_snr = PitWrapper(cal_si_snr)
    loss, perms = pit_si_snr(labels, preds)
    return loss

class SISNRLoss(nn.Module):
    """
    Scale-Invariant Signal-to-Noise Ratio (SI-SNR) Loss.
    Standard metric for time-domain speech separation.
    """
    def __init__(self, eps=1e-8):
        super(SISNRLoss, self).__init__()
        self.eps = eps

    def forward(self, targets, preds):
        """
        Args:
            preds: [Batch, Samples] or [Batch, Spk, Samples]
            targets: [Batch, Samples] or [Batch, Spk, Samples]
        Returns:
            si_snr: [Batch] or [Batch, Spk] (Negated for minimization)
        """
        # Ensure zero mean
        preds_mean = preds - torch.mean(preds, dim=-1, keepdim=True)
        targets_mean = targets - torch.mean(targets, dim=-1, keepdim=True)

        # Calculate projection
        # <s, s_hat>
        dot_product = torch.sum(preds_mean * targets_mean, dim=-1, keepdim=True)
        # ||s||^2
        target_energy = torch.sum(targets_mean ** 2, dim=-1, keepdim=True) + self.eps
        
        # Projection of prediction onto target
        projection = (dot_product / target_energy) * targets_mean

        # Noise component (orthogonal to target)
        noise = preds_mean - projection

        # Calculate SI-SNR
        # ratio = ||s_target||^2 / ||e_noise||^2
        ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + self.eps)
        
        si_snr = 10 * torch.log10(ratio + self.eps)

        # Return negative SI-SNR because we want to maximize SNR (minimize Loss)
        return -si_snr


class PITLossWrapper(nn.Module):
    """
    Permutation Invariant Training (PIT) Wrapper.
    Automatically finds the best permutation of speakers that minimizes the loss.
    """
    def __init__(self, loss_func, spk_num):
        super(PITLossWrapper, self).__init__()
        self.loss_func = loss_func
        self.spk_num = spk_num
        
        # Generate all possible permutations (e.g., for 2 spk: [(0,1), (1,0)])
        self.permutations = list(itertools.permutations(range(spk_num)))

    def forward(self, targets, preds):
        """
        Args:
            preds: [Batch, Spk, Samples]
            targets: [Batch, Spk, Samples]
        Returns:
            min_loss: Scalar (average loss over batch)
        """
        batch_size = preds.shape[0]
        
        # 1. Calculate loss for all possible permutations
        # We want to construct a matrix of shape [Batch, Permutations]
        loss_permutations = []

        for p in self.permutations:
            # p is a tuple, e.g., (1, 0)
            # Reorder targets according to this permutation
            permuted_targets = targets[:, p, :]
            
            # Calculate loss for this specific permutation
            # Result shape: [Batch, Spk]
            loss_per_spk = self.loss_func(permuted_targets, preds)
            
            # Average over speakers to get loss per batch item
            # Result shape: [Batch]
            loss_per_batch = torch.mean(loss_per_spk, dim=1)
            loss_permutations.append(loss_per_batch)

        # Stack to shape [Batch, Num_Permutations]
        loss_mat = torch.stack(loss_permutations, dim=1)

        # 2. Find the minimum loss for each batch item (the best permutation)
        min_loss, _ = torch.min(loss_mat, dim=1)

        # 3. Return the average loss over the batch
        return torch.mean(min_loss)

def get_loss(spk_num=2):
    """
    Factory function to get the ready-to-use loss module.
    """
    base_loss = SISNRLoss()
    return PITLossWrapper(base_loss, spk_num=spk_num)

class PitWrapper(nn.Module):
    """
    Permutation Invariant Wrapper to allow Permutation Invariant Training
    (PIT) with existing losses.

    Permutation invariance is calculated over the sources/classes axis which is
    assumed to be the rightmost dimension: predictions and targets tensors are
    assumed to have shape [batch, ..., channels, sources].

    Arguments
    ---------
    base_loss : function
        Base loss function, e.g. torch.nn.MSELoss. It is assumed that it takes
        two arguments:
        predictions and targets and no reduction is performed.
        (if a pytorch loss is used, the user must specify reduction="none").

    Returns
    ---------
    pit_loss : torch.nn.Module
        Torch module supporting forward method for PIT.

    Example
    -------
    >>> pit_mse = PitWrapper(nn.MSELoss(reduction="none"))
    >>> targets = torch.rand((2, 32, 4))
    >>> p = (3, 0, 2, 1)
    >>> predictions = targets[..., p]
    >>> loss, opt_p = pit_mse(predictions, targets)
    >>> loss
    tensor([0., 0.])
    """

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        """
        Arguments
        ----------
        loss_mat : torch.Tensor
            Tensor of shape [sources, source] containing loss values for each
            possible permutation of predictions.

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current batch, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.
        """

        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        """
        Arguments
        ---------
        pred : torch.Tensor
            Network prediction for the current example, tensor of
            shape [..., sources].
        target : torch.Tensor
            Target for the current example, tensor of shape [..., sources].

        Returns
        -------
        loss : torch.Tensor
            Permutation invariant loss for the current example, tensor of shape [1]

        assigned_perm : tuple
            Indexes for optimal permutation of the input over sources which
            minimizes the loss.

        """

        n_sources = pred.size(-1)

        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )

        loss_mat = self.base_loss(pred, target)
        assert (
            len(loss_mat.shape) >= 2
        ), "Base loss should not perform any reduction operation"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])

        return self._fast_pit(loss_mat)


    def forward(self, preds, targets):
        """
            Arguments
            ---------
            preds : torch.Tensor
                Network predictions tensor, of shape
                [batch, channels, ..., sources].
            targets : torch.Tensor
                Target tensor, of shape [batch, channels, ..., sources].

            Returns
            -------
            loss : torch.Tensor
                Permutation invariant loss for current examples, tensor of
                shape [batch]

            perms : list
                List of indexes for optimal permutation of the inputs over
                sources.
                e.g., [(0, 1, 2), (2, 1, 0)] for three sources and 2 examples
                per batch.
        """
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        loss = torch.stack(losses)
        return loss, perms


def cal_si_snr(source, estimate_source):
    """Calculate SI-SNR.

    Arguments:
    ---------
    source: [T, B, C],
        Where B is batch size, T is the length of the sources, C is the number of sources
        the ordering is made so that this loss is compatible with the class PitWrapper.

    estimate_source: [T, B, C]
        The estimated source.

    Example:
    ---------
    >>> import numpy as np
    >>> x = torch.Tensor([[1, 0], [123, 45], [34, 5], [2312, 421]])
    >>> xhat = x[:, (1, 0)]
    >>> x = x.unsqueeze(-1).repeat(1, 1, 2)
    >>> xhat = xhat.unsqueeze(1).repeat(1, 2, 1)
    >>> si_snr = -cal_si_snr(x, xhat)
    >>> print(si_snr)
    tensor([[[ 25.2142, 144.1789],
             [130.9283,  25.2142]]])
    """
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    device = estimate_source.device.type

    ##[32000, 2, 2]
    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1], device=device
    )
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    num_samples = (
        source_lengths.contiguous().reshape(1, -1, 1).float()
    )  # [1, B, 1]
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = (
        torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    )
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = zero_mean_target  # [T, B, C]
    s_estimate = zero_mean_estimate  # [T, B, C]
    # s_target = <s', s>s / ||s||^2
    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)  # [1, B, C]
    s_target_energy = (
        torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    )  # [1, B, C]
    proj = dot * s_target / s_target_energy  # [T, B, C]
    # e_noise = s' - s_target
    e_noise = s_estimate - proj  # [T, B, C]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    #print('proj power: {}'.format(torch.sum(proj ** 2, dim=0)))
    #print('e_noise power: {}'.format(torch.sum(e_noise ** 2, dim=0)))
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (
        torch.sum(e_noise ** 2, dim=0) + EPS
    )
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)  # [B, C]
    #print('si_snr: {}'.format(si_snr))

    #si_snr[si_snr>=80.0]  = 0.0
    return -si_snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """
    Arguments
    ---------
    source : [T, B, C]
    source_lengths : [B]

    Returns
    -------
    mask : [T, B, 1]

    Example:
    ---------
    >>> source = torch.randn(4, 3, 2)
    >>> source_lengths = torch.Tensor([2, 1, 4]).int()
    >>> mask = get_mask(source, source_lengths)
    >>> print(mask)
    tensor([[[1.],
             [1.],
             [1.]],
    <BLANKLINE>
            [[1.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]],
    <BLANKLINE>
            [[0.],
             [0.],
             [1.]]])
    """
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask

def cal_SISNR(targets, preds):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """
    preds_mean = preds - torch.mean(preds, dim=-1, keepdim=True)
    targets_mean = targets - torch.mean(targets, dim=-1, keepdim=True)

    # Calculate projection
    # <s, s_hat>
    dot_product = torch.sum(preds_mean * targets_mean, dim=-1, keepdim=True)
    # ||s||^2
    target_energy = torch.sum(targets_mean ** 2, dim=-1, keepdim=True) + EPS
    
    # Projection of prediction onto target
    projection = (dot_product / target_energy) * targets_mean

    # Noise component (orthogonal to target)
    noise = preds_mean - projection

    # Calculate SI-SNR
    # ratio = ||s_target||^2 / ||e_noise||^2
    ratio = torch.sum(projection ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + EPS)
    
    si_snr = 10 * torch.log10(ratio + EPS)

    # Return negative SI-SNR because we want to maximize SNR (minimize Loss)
    return -si_snr


def cal_SDR(target, est_target):
    assert target.size() == est_target.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(target, dim=1, keepdim=True)
    mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
    target = target - mean_source
    est_target = est_target - mean_estimate
    # Step 2. SDR
    scaled_target = target
    e_noise = est_target - target
    sdr = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
    sdr = 10 * torch.log10(sdr + EPS)
    return -sdr  



class SpeechSeparationLoss(_Loss):
    """Loss wrapper for speech separation tasks"""

    def __init__(self, loss_type='pit_sisnr'):
        super(SpeechSeparationLoss, self).__init__()
        self.loss_type = loss_type

        # For hybrid loss
        if 'hybrid' in self.loss_type:
            self.stft_loss = MultiResolutionSTFTLoss()
        

    def forward(self, clean, estimate):
        """
        Args:
            clean: [B, C, T] - target sources for C speakers
            estimate: [B, C, T] - estimated sources for C speakers

        Returns:
            loss: scalar loss value
        """
        if self.loss_type == 'snr':
            # SDR loss
            loss = torch.mean(cal_SDR(clean, estimate))
        elif self.loss_type == 'sisdr':
            # SI-SDR loss (note: sisdr typically means si-snr)
            loss = torch.mean(cal_SISNR(clean, estimate))
        elif self.loss_type == 'pit_sisnr':
            # Permutation Invariant Training with SI-SNR
            loss = self.pit_loss(clean, estimate, loss_fn='sisnr')
        elif self.loss_type == 'pit_sdr':
            # Permutation Invariant Training with SDR
            loss = self.pit_loss(clean, estimate, loss_fn='sdr')
        elif self.loss_type == 'pit_hybrid':
            # Hybrid loss with PIT
            loss = self.pit_loss(clean, estimate, loss_fn='hybrid')
        elif self.loss_type == 'loss_moss':
            # print( "AAAAAAAAAAAAAA",clean.transpose(1,2).shape,estimate.transpose(1,2).shape)
            loss = loss_moss(clean,estimate)
        else:
            raise NameError(f'Unsupported loss type: {self.loss_type}')

        return loss

    def pit_loss(self, clean, estimate, loss_fn='sisnr'):
        """
        Permutation Invariant Training loss for speech separation

        Args:
            clean: [B, C, T] - target sources for C speakers
            estimate: [B, C, T] - estimated sources for C speakers
            loss_fn: 'sisnr', 'sdr', or 'hybrid'

        Returns:
            loss: scalar PIT loss
        """
        assert clean.shape == estimate.shape, f"Shape mismatch: clean {clean.shape}, estimate {estimate.shape}"
        if clean.dim() == 4:
            batch_size, num_speakers, num_channels, time_len = clean.shape
            if num_channels != 2:
                raise ValueError(f"stereo loss expects 2 channels, got {num_channels}")
        else:
            batch_size, num_speakers, time_len = clean.shape

        # Generate all possible permutations
        import itertools
        perms = list(itertools.permutations(range(num_speakers)))

        losses = []
        for perm in perms:
            # Rearrange estimates according to permutation
            if clean.dim() == 4:
                reordered_clean = clean[:, perm, :, :]
            else:
                reordered_clean = clean[:, perm, :]

            # Calculate loss for this permutation
            if loss_fn == 'sisnr':
                perm_loss = cal_SISNR(reordered_clean, estimate)
                if clean.dim() == 4:
                    # Average over channels, then over speakers
                    perm_loss = torch.mean(perm_loss, dim=2)
                perm_loss = torch.mean(perm_loss, dim=1)
            elif loss_fn == 'sdr':
                if clean.dim() == 4:
                    raise ValueError("SDR loss does not support multi-channel inputs")
                perm_loss = torch.mean(cal_SDR(reordered_clean, estimate))
            elif loss_fn == 'hybrid':
                if clean.dim() == 4:
                    raise ValueError("Hybrid loss does not support multi-channel inputs")
                perm_loss = torch.mean(cal_SISNR(reordered_clean, estimate))
                if hasattr(self, 'stft_loss'):
                    # Add STFT loss for each speaker
                    stft_loss = 0
                    for spk in range(num_speakers):
                        stft_loss += self.stft_loss(reordered_clean[:, spk, :], estimate[:, spk, :])
                    perm_loss += stft_loss / num_speakers
            else:
                raise ValueError(f"Unknown loss function: {loss_fn}")
            # print(perm_loss)
            losses.append(perm_loss)

        # Stack losses and find minimum
        losses = torch.stack(losses, dim=1)  # [num_perms]
        min_loss, best_perm_idx = torch.min(losses, dim=1)

        return torch.mean(min_loss)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for speech enhancement/separation"""

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240]):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths

    def forward(self, clean, estimate):
        """
        Args:
            clean: [B, T] - clean signal
            estimate: [B, T] - estimated signal

        Returns:
            loss: scalar STFT loss
        """
        loss = 0
        for fft_size, hop_size, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # STFT
            clean_spec = torch.stft(clean, n_fft=fft_size, hop_length=hop_size,
                                  win_length=win_length, return_complex=True)
            estimate_spec = torch.stft(estimate, n_fft=fft_size, hop_length=hop_size,
                                     win_length=win_length, return_complex=True)

            # Magnitude and phase
            clean_mag = torch.abs(clean_spec)
            estimate_mag = torch.abs(estimate_spec)

            # Log magnitude loss
            loss += F.l1_loss(torch.log(clean_mag + EPS), torch.log(estimate_mag + EPS))

        return loss


def get_loss_function(loss_type='pit_sisnr'):
    """Factory function to get loss function instance

    Args:
        loss_type: str, type of loss function
            - 'snr': Signal-to-Noise Ratio
            - 'sisdr': Scale-Invariant Signal-to-Distortion Ratio
            - 'pit_sisnr': PIT with SI-SNR (default)
            - 'pit_sdr': PIT with SDR
            - 'pit_hybrid': PIT with hybrid loss

    Returns:
        loss_fn: SpeechSeparationLoss instance
    """
    return SpeechSeparationLoss(loss_type=loss_type)
