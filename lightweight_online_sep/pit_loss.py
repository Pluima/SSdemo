import itertools

import torch


def _si_snr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-Invariant SNR for tensors with shape [B, T]."""
    est = est - torch.mean(est, dim=-1, keepdim=True)
    ref = ref - torch.mean(ref, dim=-1, keepdim=True)

    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref
    proj = proj / (torch.sum(ref * ref, dim=-1, keepdim=True) + eps)

    noise = est - proj
    ratio = torch.sum(proj * proj, dim=-1) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def pit_si_snr_loss(est_sources: torch.Tensor, ref_sources: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    PIT SI-SNR loss.

    Args:
        est_sources: [B, S, T]
        ref_sources: [B, S, T]
    Returns:
        scalar loss (negative SI-SNR)
    """
    if est_sources.shape != ref_sources.shape:
        raise ValueError(
            f"Shape mismatch: est={tuple(est_sources.shape)} ref={tuple(ref_sources.shape)}"
        )
    if est_sources.ndim != 3:
        raise ValueError("Expected [B, S, T] tensors")

    b, s, _ = est_sources.shape

    if s == 2:
        s00 = _si_snr(est_sources[:, 0], ref_sources[:, 0], eps=eps)
        s11 = _si_snr(est_sources[:, 1], ref_sources[:, 1], eps=eps)
        s01 = _si_snr(est_sources[:, 0], ref_sources[:, 1], eps=eps)
        s10 = _si_snr(est_sources[:, 1], ref_sources[:, 0], eps=eps)

        pair_a = (s00 + s11) * 0.5
        pair_b = (s01 + s10) * 0.5
        best = torch.maximum(pair_a, pair_b)
        return -torch.mean(best)

    perms = list(itertools.permutations(range(s)))
    perm_scores = []
    for perm in perms:
        score = 0.0
        for est_i, ref_i in enumerate(perm):
            score = score + _si_snr(est_sources[:, est_i], ref_sources[:, ref_i], eps=eps)
        perm_scores.append(score / float(s))

    stacked = torch.stack(perm_scores, dim=1)  # [B, P]
    best, _ = torch.max(stacked, dim=1)
    return -torch.mean(best)
