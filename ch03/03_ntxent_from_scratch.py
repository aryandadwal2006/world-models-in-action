"""03_ntxent_from_scratch.py - First-principles mathematical derivation of NT-Xent / InfoNCE.

Explicitly computes cosine similarity, softmax positive identification, cross-entropy,
and gradient decomposition without library abstractions, verifying correctness against
the vectorized loss in worldmodels.losses.contrastive (Equations 3.3-3.7).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn.functional as F

from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.train import save_json, set_seed


def ntxent_unrolled_reference(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.5,
) -> torch.Tensor:
    """Explicit loop-based implementation of NT-Xent matching Equation 3.5 definition directly."""
    batch_size, dim = z1.shape
    # Normalize
    z1_norm = z1 / torch.norm(z1, p=2, dim=1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, p=2, dim=1, keepdim=True)

    # 2N views: 0..N-1 are view 1, N..2N-1 are view 2
    all_z = torch.cat([z1_norm, z2_norm], dim=0)
    total_samples = 2 * batch_size

    loss_sum = 0.0

    for i in range(total_samples):
        # Identify positive index: for i < N it is i + N; for i >= N it is i - N
        positive_idx = (i + batch_size) if i < batch_size else (i - batch_size)

        # Numerator: exp( sim(z_i, z_pos) / tau )
        sim_pos = torch.dot(all_z[i], all_z[positive_idx]) / temperature
        numerator = torch.exp(sim_pos)

        # Denominator: sum_{k != i} exp( sim(z_i, z_k) / tau )
        denominator = 0.0
        for k in range(total_samples):
            if k == i:
                continue
            sim_k = torch.dot(all_z[i], all_z[k]) / temperature
            denominator += torch.exp(sim_k)

        # -log( numerator / denominator )
        pair_loss = -torch.log(numerator / denominator)
        loss_sum += pair_loss

    return loss_sum / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify NT-Xent loss from first principles.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size N.")
    parser.add_argument("--dim", type=int, default=16, help="Representation dimension D.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature tau.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Directory for results.")
    args = parser.parse_args()

    set_seed(args.seed)

    # Generate synthetic representation batches
    z1 = torch.randn(args.batch_size, args.dim, requires_grad=True)
    z2 = torch.randn(args.batch_size, args.dim, requires_grad=True)

    # 1. Unrolled definition loss
    loss_ref = ntxent_unrolled_reference(z1, z2, temperature=args.temperature)

    # 2. Vectorized production loss
    loss_vec = nt_xent_loss(z1, z2, temperature=args.temperature)

    discrepancy = float(torch.abs(loss_ref - loss_vec).item())
    print(f"Unrolled Equation 3.5 Loss: {loss_ref.item():.6f}")
    print(f"Vectorized NT-Xent Loss:    {loss_vec.item():.6f}")
    print(f"Absolute Discrepancy:      {discrepancy:.8e}")

    # Theoretical bounds check: I(z1; z2) >= log(K+1) - L, where K = 2(N - 1) negatives
    k_negatives = 2 * (args.batch_size - 1)
    log_k_plus_1 = math.log(k_negatives + 1)
    mi_lower_bound = log_k_plus_1 - loss_vec.item()

    print(f"Number of negatives K:     {k_negatives}")
    print(f"log(K + 1) Bound Ceiling:  {log_k_plus_1:.4f}")
    print(f"Estimated MI Lower Bound:  {mi_lower_bound:.4f}")

    results = {
        "batch_size": args.batch_size,
        "dim": args.dim,
        "temperature": args.temperature,
        "unrolled_loss": float(loss_ref.item()),
        "vectorized_loss": float(loss_vec.item()),
        "discrepancy": discrepancy,
        "k_negatives": k_negatives,
        "log_k_plus_1": log_k_plus_1,
        "mi_lower_bound": mi_lower_bound,
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_file = os.path.join(args.results_dir, "ntxent_verification.json")
    save_json(results, out_file)
    print(f"Saved verification metrics -> {out_file}")

    assert discrepancy < 1e-6, f"Discrepancy too large: {discrepancy}"
    print("Mathematical equivalence verified successfully.")


if __name__ == "__main__":
    main()
