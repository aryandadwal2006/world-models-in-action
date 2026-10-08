"""Contrastive, non-contrastive, and redundancy-reduction self-supervised objectives.

Implements exact mathematical definitions for:
- NT-Xent / InfoNCE loss (Equations 3.3-3.7)
- SimSiam negative cosine loss with stop-gradient (Equations 3.9-3.10)
- BYOL exponential moving average target update (Equation 3.11)
- Barlow Twins cross-correlation loss (Equations 3.12-3.13)
- VICReg variance-invariance-covariance loss (Equations 3.14-3.17)
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def nt_xent_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Computes the normalized temperature-scaled cross-entropy (NT-Xent) loss.

    Given a batch of N samples, two stochastic views produce representations z1, z2.
    For each anchor view, the remaining view from the same sample forms the sole positive,
    while all other 2(N - 1) representations in the batch serve as negative distractors.

    Equations:
        sim(u, v) = (u^T v) / (||u||_2 * ||v||_2)  [Eq 3.3]
        L_{i, j} = -log( exp(sim(z_i, z_j) / tau) / sum_{k != i} exp(sim(z_i, z_k) / tau) )  [Eq 3.5]

    Args:
        z1: Tensor of shape (N, D) representing representations from view 1.
        z2: Tensor of shape (N, D) representing representations from view 2.
        temperature: Temperature hyperparameter tau scaling logits.

    Returns:
        Scalar NT-Xent loss averaged over all 2N positive pairs.
    """
    batch_size = z1.shape[0]
    device = z1.device

    # L2 normalize representations along feature dimension
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)

    # Concatenate views into unified batch of size 2N
    representations = torch.cat([z1_norm, z2_norm], dim=0)  # (2N, D)

    # Compute pairwise cosine similarity matrix scaled by temperature
    similarity_matrix = torch.matmul(representations, representations.T) / temperature  # (2N, 2N)

    # Mask out diagonal (self-similarity)
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    # Masked elements set to large negative value so exp(-inf) -> 0 in softmax denominator
    similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

    # For index i in [0, N-1], positive is i + N.
    # For index i in [N, 2N-1], positive is i - N.
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device),
    ], dim=0)

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


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
