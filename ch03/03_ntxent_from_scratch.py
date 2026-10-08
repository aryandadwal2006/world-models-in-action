"""03_ntxent_from_scratch.py - Verify the NT-Xent implementation.

The script checks the explicit pairwise definition against the vectorized
implementation, verifies that their gradients agree, and reports the nominal
InfoNCE lower-bound quantity under the sampling convention used in the chapter.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.train import save_json, set_seed


def ntxent_unrolled_reference(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Direct loop-based implementation of the NT-Xent definition."""
    if z1.ndim != 2 or z2.ndim != 2 or z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape (N, D)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    z1_norm = z1 / torch.linalg.vector_norm(z1, dim=1, keepdim=True)
    z2_norm = z2 / torch.linalg.vector_norm(z2, dim=1, keepdim=True)
    all_z = torch.cat([z1_norm, z2_norm], dim=0)
    n = z1.shape[0]

    losses = []
    for i in range(2 * n):
        positive_idx = i + n if i < n else i - n
        logits = []
        for k in range(2 * n):
            if k == i:
                continue
            logits.append(torch.dot(all_z[i], all_z[k]) / temperature)
        logits = torch.stack(logits)
        positive_position = positive_idx if positive_idx < i else positive_idx - 1
        losses.append(-logits[positive_position] + torch.logsumexp(logits, dim=0))

    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NT-Xent from first principles.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    args = parser.parse_args()

    set_seed(args.seed)

    z1_ref = torch.randn(args.batch_size, args.dim, requires_grad=True)
    z2_ref = torch.randn(args.batch_size, args.dim, requires_grad=True)
    z1_vec = z1_ref.detach().clone().requires_grad_(True)
    z2_vec = z2_ref.detach().clone().requires_grad_(True)

    loss_ref = ntxent_unrolled_reference(z1_ref, z2_ref, args.temperature)
    loss_vec = nt_xent_loss(z1_vec, z2_vec, args.temperature)

    grad_ref = torch.autograd.grad(loss_ref, z1_ref, retain_graph=False)[0]
    grad_vec = torch.autograd.grad(loss_vec, z1_vec, retain_graph=False)[0]

    loss_discrepancy = float(torch.abs(loss_ref - loss_vec).item())
    grad_discrepancy = float(torch.max(torch.abs(grad_ref - grad_vec)).item())

    k_negatives = 2 * (args.batch_size - 1)
    nominal_bound = math.log(k_negatives + 1) - float(loss_vec.item())

    print(f"Unrolled loss:       {loss_ref.item():.8f}")
    print(f"Vectorized loss:     {loss_vec.item():.8f}")
    print(f"Loss discrepancy:    {loss_discrepancy:.3e}")
    print(f"Max gradient delta:  {grad_discrepancy:.3e}")
    print(f"K negatives:         {k_negatives}")
    print(f"Nominal log(K+1)-L:  {nominal_bound:.6f}")

    if loss_discrepancy >= 1e-6 or grad_discrepancy >= 1e-6:
        raise AssertionError("Reference and vectorized NT-Xent implementations disagree")

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "batch_size": args.batch_size,
            "dim": args.dim,
            "temperature": args.temperature,
            "unrolled_loss": float(loss_ref.item()),
            "vectorized_loss": float(loss_vec.item()),
            "loss_discrepancy": loss_discrepancy,
            "max_gradient_discrepancy": grad_discrepancy,
            "k_negatives": k_negatives,
            "nominal_log_k_plus_1_minus_loss": nominal_bound,
        },
        os.path.join(args.results_dir, "ntxent_verification.json"),
    )


if __name__ == "__main__":
    main()
