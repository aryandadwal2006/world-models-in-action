"""Non-contrastive and redundancy-reduction learning objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def simsiam_loss(
    p1: torch.Tensor,
    z2: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
) -> torch.Tensor:
    """Computes the symmetric SimSiam negative cosine similarity loss with stop-gradient.

    Equations:
        D(p_1, stopgrad(z_2)) = - (p_1 / ||p_1||_2) . (z_2 / ||z_2||_2)  [Eq 3.9]
        L = 0.5 * D(p_1, stopgrad(z_2)) + 0.5 * D(p_2, stopgrad(z_1))     [Eq 3.10]

    Args:
        p1: Output from predictor head on view 1, shape (N, D).
        z2: Target representation from encoder on view 2, shape (N, D).
        p2: Output from predictor head on view 2, shape (N, D).
        z1: Target representation from encoder on view 1, shape (N, D).

    Returns:
        Scalar symmetric negative cosine similarity loss.
    """
    def _d(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        p_norm = F.normalize(p, dim=1)
        z_norm = F.normalize(z.detach(), dim=1)
        return -(p_norm * z_norm).sum(dim=1).mean()

    return 0.5 * _d(p1, z2) + 0.5 * _d(p2, z1)


def update_target_ema(
    online_model: nn.Module,
    target_model: nn.Module,
    decay: float = 0.99,
) -> None:
    """Updates target model parameters via Exponential Moving Average (EMA).

    Equation:
        theta_{target} <- tau * theta_{target} + (1 - tau) * theta_{online}  [Eq 3.11]

    Args:
        online_model: Source network providing active gradient updates.
        target_model: Slowly moving target network.
        decay: Momentum decay parameter tau in [0, 1].
    """
    with torch.no_grad():
        for param_online, param_target in zip(online_model.parameters(), target_model.parameters()):
            param_target.data.mul_(decay).add_(param_online.data, alpha=1.0 - decay)


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambd: float = 0.005,
) -> torch.Tensor:
    """Computes Barlow Twins cross-correlation redundancy-reduction loss.

    Equations:
        C_{i, j} = sum_b(z_{b, i}^A * z_{b, j}^B) / (sqrt(sum_b (z_{b, i}^A)^2) * sqrt(sum_b (z_{b, j}^B)^2))  [Eq 3.12]
        L = sum_i (1 - C_{i, i})^2 + lambda * sum_i sum_{j != i} C_{i, j}^2  [Eq 3.13]

    Args:
        z1: First batch of embeddings, shape (N, D).
        z2: Second batch of embeddings, shape (N, D).
        lambd: Off-diagonal penalty weight lambda.

    Returns:
        Scalar Barlow Twins loss.
    """
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape (N, D)")
    if z1.shape[0] < 2:
        raise ValueError("Barlow Twins requires batch_size >= 2")
    if lambd < 0:
        raise ValueError("lambd must be non-negative")

    batch_size, dim = z1.shape

    # Use population standard deviation so that the normalized diagonal
    # correlation is 1 when the two views are identical.
    std1 = z1.std(dim=0, unbiased=False).clamp_min(1e-6)
    std2 = z2.std(dim=0, unbiased=False).clamp_min(1e-6)
    z1_norm = (z1 - z1.mean(dim=0)) / std1
    z2_norm = (z2 - z2.mean(dim=0)) / std2

    # Cross-correlation matrix C of shape (D, D)
    cross_corr = torch.matmul(z1_norm.T, z2_norm) / batch_size

    # Loss: diagonal terms pushed to 1, off-diagonal terms pushed to 0
    on_diag = torch.diagonal(cross_corr).add(-1.0).pow(2).sum()

    # Off-diagonal mask
    off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=z1.device)
    off_diag = cross_corr[off_diag_mask].pow(2).sum()

    return on_diag + lambd * off_diag
