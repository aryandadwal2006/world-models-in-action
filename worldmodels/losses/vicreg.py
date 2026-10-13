"""Variance-Invariance-Covariance Regularization (VICReg)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes Variance-Invariance-Covariance Regularization (VICReg) loss.

    Equations:
        s(Z^A, Z^B) = (1/n) * sum_b ||z_b^A - z_b^B||_2^2                     [Eq 3.15, invariance]
        v(Z) = (1/d) * sum_j max(0, gamma - sqrt(Var(z_j) + eps))             [Eq 3.14, variance hinge]
        c(Z) = (1/d) * sum_{i != j} [Cov(Z)]_{i, j}^2                         [Eq 3.16, covariance]
        L = sim_coeff * s + std_coeff * (v(Z^A) + v(Z^B)) + cov_coeff * (c(Z^A) + c(Z^B))  [Eq 3.17]

    Args:
        z_a: Embeddings from branch A, shape (N, D).
        z_b: Embeddings from branch B, shape (N, D).
        sim_coeff: Weight for invariance loss.
        std_coeff: Weight for variance hinge loss.
        cov_coeff: Weight for covariance decorrelation loss.
        gamma: Target standard deviation threshold.
        eps: Small positive scalar for numerical stability in sqrt.

    Returns:
        Tuple of (total_loss, invariance_loss, variance_loss, covariance_loss).
    """
    n, d = z_a.shape

    # 1. Invariance term: mean squared Euclidean distance
    sim_loss = F.mse_loss(z_a, z_b)

    # 2. Variance hinge term: ensures std >= gamma for each dimension
    def _variance_loss(z: torch.Tensor) -> torch.Tensor:
        # Regularized standard deviation per dimension
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        # Hinge loss against target gamma
        return F.relu(gamma - std).mean()

    var_loss = _variance_loss(z_a) + _variance_loss(z_b)

    # 3. Covariance term: penalizes correlation between distinct dimensions
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (n - 1)
        # Off-diagonal elements squared
        diag_mask = ~torch.eye(d, dtype=torch.bool, device=z.device)
        return cov[diag_mask].pow(2).sum() / d

    cov_loss = _covariance_loss(z_a) + _covariance_loss(z_b)

    total_loss = sim_coeff * sim_loss + std_coeff * var_loss + cov_coeff * cov_loss
    return total_loss, sim_loss, var_loss, cov_loss
