from __future__ import annotations

import torch


def sample_token_mask(
    eligible_mask: torch.Tensor,
    mask_ratio: float,
    *,
    min_masked_tokens: int = 1,
    generator: torch.Generator | None = None,
) -> torch.BoolTensor:
    """
    Randomly sample token masks from an eligibility mask.

    Args:
        eligible_mask: bool tensor of shape `(B, L)` where True denotes a token
            that may be masked.
        mask_ratio: probability of masking an eligible token.
    """

    if eligible_mask.dtype != torch.bool:
        raise ValueError("eligible_mask must be boolean.")
    if eligible_mask.ndim != 2:
        raise ValueError(
            f"eligible_mask must have shape (B, L), got {tuple(eligible_mask.shape)}"
        )
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must lie in [0, 1], got {mask_ratio}.")

    sampled = torch.zeros_like(eligible_mask)
    if mask_ratio == 0.0:
        return sampled

    for batch_idx in range(eligible_mask.shape[0]):
        candidates = torch.nonzero(eligible_mask[batch_idx], as_tuple=False).flatten()
        if candidates.numel() == 0:
            continue
        random_values = torch.rand(
            candidates.numel(),
            device=eligible_mask.device,
            generator=generator,
        )
        chosen = candidates[random_values < mask_ratio]
        if chosen.numel() < min_masked_tokens:
            perm = torch.randperm(
                candidates.numel(),
                device=eligible_mask.device,
                generator=generator,
            )
            chosen = candidates[perm[: min(min_masked_tokens, candidates.numel())]]
        sampled[batch_idx, chosen] = True
    return sampled
