"""NT-Xent / InfoNCE objective for paired visual views."""

from __future__ import annotations

import torch
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
