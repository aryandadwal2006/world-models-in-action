"""Reconstruction loss functions for generative representation learning.

Implements masked mean squared error (MSE) on masked patches (Equation 3.18)
for Masked Autoencoder (MAE) architectures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mse_loss(
    target_patches: torch.Tensor,
    predicted_patches: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Computes mean squared error strictly over masked visual patches.

    Equation:
        L_MAE = (1 / sum(M)) * sum_{b, i} M_{b, i} * ||P_{b, i} - \hat{P}_{b, i}||_2^2  [Eq 3.18]

    Args:
        target_patches: Ground-truth unmasked patches of shape (B, N_patches, patch_dim).
        predicted_patches: Reconstructed patches of shape (B, N_patches, patch_dim).
        mask: Binary mask of shape (B, N_patches), where 1 indicates a masked patch
            and 0 indicates an unmasked (visible) patch.

    Returns:
        Scalar mean squared error computed solely across masked patch tokens.
    """
    # Squared error per patch
    squared_errors = F.mse_loss(predicted_patches, target_patches, reduction="none")  # (B, N, patch_dim)
    per_patch_loss = squared_errors.mean(dim=-1)  # (B, N)

    # Average loss over all masked patches
    mask_bool = mask.bool()
    if mask_bool.sum() == 0:
        return torch.tensor(0.0, device=target_patches.device)

    loss = per_patch_loss[mask_bool].mean()
    return loss
