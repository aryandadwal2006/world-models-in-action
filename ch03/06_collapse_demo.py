"""06_collapse_demo.py - Empirical demonstration of representation collapse.

Demonstrates how minimizing positive attraction without repulsive negative forces,
stop-gradient boundaries, or variance constraints leads to dimensional and complete collapse:
all inputs map to a single constant point, driving latent standard deviation to zero (Figure 3.9).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate representation collapse.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dim.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def run_experiment(
    mode: str,
    seed: int,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    """Runs training either in 'attraction_only' (collapsing) or 'contrastive' (repulsion) mode."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    view_pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    epoch_stds = []

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = view_pipeline(images)

            z1 = encoder(v1)
            z2 = encoder(v2)
            p1 = proj_head(z1)
            p2 = proj_head(z2)

            if mode == "attraction_only":
                # Naive MSE attraction without negative repulsion or variance floor
                loss = F.mse_loss(p1, p2)
            else:
                loss = nt_xent_loss(p1, p2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure per-dimension standard deviation across validation / evaluation set
        encoder.eval()
        all_z = []
        with torch.no_grad():
            for batch in train_loader:
                img = batch["image"].to(device)
                all_z.append(encoder(img).cpu())
        z_concat = torch.cat(all_z, dim=0)  # (N, D)
        # Average std across latent dimensions
        mean_dim_std = float(z_concat.std(dim=0).mean().item())
        epoch_stds.append(mean_dim_std)

    return epoch_stds


def main() -> None:
    args = parse_args()
    device = get_device()

    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    train_raw = load_dataset_npz(train_file)
    n_train = min(len(train_raw["frames"]), 3000)

    train_ds = DMCDataset(
        frames=train_raw["frames"][:n_train],
        physics_states=train_raw["physics_states"][:n_train],
        episode_ids=train_raw["episode_ids"][:n_train],
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    collapse_runs = []
    contrastive_runs = []

    print(f"=== Running Collapse Demonstration across seeds {args.seeds} ===")
    for seed in args.seeds:
        print(f"Seed {seed} | Running attraction-only training (no negatives)...")
        stds_col = run_experiment("attraction_only", seed, train_loader, args, device)
        collapse_runs.append(stds_col)
        print(f"  Final latent std: {stds_col[-1]:.6f}")

        print(f"Seed {seed} | Running contrastive training (with negatives)...")
        stds_con = run_experiment("contrastive", seed, train_loader, args, device)
        contrastive_runs.append(stds_con)
        print(f"  Final latent std: {stds_con[-1]:.6f}")

    col_arr = np.array(collapse_runs)  # (seeds, epochs)
    con_arr = np.array(contrastive_runs)

    epochs = np.arange(1, args.epochs + 1)

    # Plot Figure 3.9
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.plot(epochs, np.mean(con_arr, axis=0), color="black", linestyle="-", linewidth=2.0, label="With Negatives (NT-Xent)")
    ax.fill_between(
        epochs,
        np.mean(con_arr, axis=0) - np.std(con_arr, axis=0),
        np.mean(con_arr, axis=0) + np.std(con_arr, axis=0),
        color="black",
        alpha=0.15,
    )

    ax.plot(epochs, np.mean(col_arr, axis=0), color="black", linestyle="--", linewidth=2.0, label="Attraction Only (No Negatives -> Collapse)")
    ax.fill_between(
        epochs,
        np.mean(col_arr, axis=0) - np.std(col_arr, axis=0),
        np.mean(col_arr, axis=0) + np.std(col_arr, axis=0),
        color="black",
        alpha=0.1,
    )

    ax.set_title("Representation Collapse: Latent Standard Deviation Over Training", fontsize=10, pad=8)
    ax.set_xlabel("Training Epoch", fontsize=9)
    ax.set_ylabel("Mean Per-Dimension Standard Deviation", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    fig_path = os.path.join(args.figures_dir, "fig03_09_collapse_demo.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.9 -> {fig_path}")

    res_path = os.path.join(args.results_dir, "collapse_demo.json")
    save_json({
        "seeds": args.seeds,
        "epochs": args.epochs,
        "collapse_mean_std": [float(x) for x in np.mean(col_arr, axis=0)],
        "contrastive_mean_std": [float(x) for x in np.mean(con_arr, axis=0)],
    }, res_path)
    print(f"Saved collapse metrics -> {res_path}")


if __name__ == "__main__":
    main()
