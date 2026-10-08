"""09_masked_autoencoder.py - Masked autoencoding on DMC visual observations.

Implements patch-based masked image modeling (75% masking ratio), computes
masked MSE reconstruction loss strictly over masked patches (Equation 3.18),
visualizes reconstructions (Figure 3.12), and measures state decodability via linear probe.
"""

from __future__ import annotations
from typing import Tuple

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.reconstruction import masked_mse_loss
from worldmodels.models.encoders import MAEEncoder, SimpleMAE
from worldmodels.train import get_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Masked Autoencoder.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--mask-ratio", type=float, default=0.75, help="Masking ratio.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    args = parser.parse_args()

    device = get_device()
    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    val_file = os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")

    train_raw = load_dataset_npz(train_file)
    val_raw = load_dataset_npz(val_file)

    n_train = min(len(train_raw["frames"]), 4000)
    n_val = min(len(val_raw["frames"]), 1000)

    train_ds = DMCDataset(
        frames=train_raw["frames"][:n_train],
        physics_states=train_raw["physics_states"][:n_train],
        episode_ids=train_raw["episode_ids"][:n_train],
    )
    val_ds = DMCDataset(
        frames=val_raw["frames"][:n_val],
        physics_states=val_raw["physics_states"][:n_val],
        episode_ids=val_raw["episode_ids"][:n_val],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    seed_probe_r2 = []

    print(f"=== Training Masked Autoencoder across seeds {args.seeds} ===")
    sample_vis = None

    for seed in args.seeds:
        set_seed(seed)
        model = SimpleMAE(latent_dim=args.latent_dim, mask_ratio=args.mask_ratio).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                loss, _, _ = model(images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(images)

            avg_loss = total_loss / len(train_ds)
            if epoch % 5 == 0 or epoch == args.epochs:
                print(f"Seed {seed} | Epoch {epoch:02d}/{args.epochs:02d} | Masked MSE: {avg_loss:.5f}")

        # Evaluate linear probe on frozen MAE latent representation
        probe_res = evaluate_linear_probe(MAEEncoder(model), train_loader, val_loader, device)
        seed_probe_r2.append(probe_res["mean_r2"])
        print(f"Seed {seed} | MAE Linear Probe Mean R^2: {probe_res['mean_r2']:.4f}")

        if seed == args.seeds[0]:
            # Save sample reconstructions for Figure 3.12
            model.eval()
            with torch.no_grad():
                val_batch = next(iter(val_loader))["image"][:4].to(device)
                _, pred_p, mask = model(val_batch)
                orig_p = model.patchify(val_batch)
                # Reconstructed image: visible patches kept, masked patches replaced with predictions
                recon_p = orig_p.clone()
                mask_bool = mask.bool().unsqueeze(-1).repeat(1, 1, model.patch_dim)
                recon_p[mask_bool] = pred_p[mask_bool]
                recon_imgs = model.unpatchify(recon_p).clamp(0, 1)

                # Masked input view (replace masked patches with gray 0.5)
                masked_input_p = orig_p.clone()
                masked_input_p[mask_bool] = 0.5
                masked_input_imgs = model.unpatchify(masked_input_p)

                sample_vis = (val_batch.cpu(), masked_input_imgs.cpu(), recon_imgs.cpu())

    mean_r2 = float(np.mean(seed_probe_r2))
    std_r2 = float(np.std(seed_probe_r2))
    print(f"\nMAE Aggregate Probe R^2 (3 seeds): {mean_r2:.3f} ± {std_r2:.3f}")

    # Plot Figure 3.12: Original, 75% Masked, Reconstructed
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(8.5, 6.2))
    row_titles = ["Original (100% Pixels)", "Masked Input (75% Patches Removed)", "MAE Reconstruction"]

    for col in range(4):
        orig, masked, recon = sample_vis
        axes[0, col].imshow(orig[col].permute(1, 2, 0).numpy())
        axes[1, col].imshow(masked[col].permute(1, 2, 0).numpy())
        axes[2, col].imshow(recon[col].permute(1, 2, 0).numpy())

    for row_idx, r_title in enumerate(row_titles):
        axes[row_idx, 0].set_ylabel(r_title, fontsize=9)
        for col in range(4):
            axes[row_idx, col].set_xticks([])
            axes[row_idx, col].set_yticks([])

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_12_mae_reconstruction.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.12 -> {fig_path}")

    out_json = os.path.join(args.results_dir, "mae_metrics.json")
    save_json({
        "seeds": args.seeds,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "seed_probe_r2": seed_probe_r2,
    }, out_json)
    print(f"Saved MAE results -> {out_json}")


if __name__ == "__main__":
    main()
