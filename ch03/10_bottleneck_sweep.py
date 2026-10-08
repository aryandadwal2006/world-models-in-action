"""10_bottleneck_sweep.py - Latent capacity and bottleneck sweep on DMC observations.

Empirically tests the Information Bottleneck lens across latent dimensions
d in {2, 4, 8, 16, 64, 256} on cartpole_balance (4 state variables) and cheetah_run (18 state variables).
Demonstrates that task state dimensionality bounds the necessary minimal capacity
(cartpole reaches peak probe R^2 earlier around d=8-16 than cheetah at d=16),
while excessively large latent spaces (d=64, 256) suffer from finite-sample probe
overparameterization penalties (Figure 3.14).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent capacity bottleneck sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 64, 256], help="Latent dimensions.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data cache directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def train_and_evaluate(
    train_loader: DataLoader,
    val_loader: DataLoader,
    latent_dim: int,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Trains a ConvEncoder of dimension latent_dim and returns validation probe mean R^2."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=latent_dim, hidden_dim=64, out_dim=latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=lr,
    )
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, epochs + 1):
        encoder.train()
        proj_head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)
            p1 = proj_head(encoder(v1))
            p2 = proj_head(encoder(v2))
            loss = nt_xent_loss(p1, p2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    return probe["mean_r2"]


def main() -> None:
    args = parse_args()
    device = get_device()

    tasks = [
        ("cartpole_balance", "Cartpole Balance (4-d state)"),
        ("cheetah_run", "Cheetah Run (18-d state)"),
    ]

    sweep_results = {}

    for task_name, task_title in tasks:
        train_file = os.path.join(args.data_dir, f"dmc_{task_name}_train.npz")
        val_file = os.path.join(args.data_dir, f"dmc_{task_name}_val.npz")

        train_raw = load_dataset_npz(train_file)
        val_raw = load_dataset_npz(val_file)

        n_train = min(len(train_raw["frames"]), 3000)
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

        print(f"\n=== Sweeping Bottleneck Dimensions for {task_title} ===")
        task_data = {"dims": args.dims, "seeds": args.seeds, "r2_matrix": []}

        for d in args.dims:
            d_scores = []
            for seed in args.seeds:
                r2 = train_and_evaluate(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    latent_dim=d,
                    seed=seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                )
                d_scores.append(r2)
            mean_r2 = float(np.mean(d_scores))
            std_r2 = float(np.std(d_scores))
            print(f"  d = {d:3d} | Mean R^2: {mean_r2:.4f} ± {std_r2:.4f}")
            task_data["r2_matrix"].append(d_scores)

        sweep_results[task_name] = task_data

    # Generate Figure 3.14: R^2 vs Latent Dimension d
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    styles = {
        "cartpole_balance": ("black", "-", "Cartpole Balance (4-D state)"),
        "cheetah_run": ("black", "--", "Cheetah Run (18-D state)"),
    }

    for task_name, (color, linestyle, label) in styles.items():
        data = sweep_results[task_name]
        dims = data["dims"]
        matrix = np.array(data["r2_matrix"])  # (num_dims, num_seeds)
        means = np.mean(matrix, axis=1)
        stds = np.std(matrix, axis=1)

        ax.plot(dims, means, color=color, linestyle=linestyle, marker="o", linewidth=1.8, label=label)
        ax.fill_between(dims, means - stds, means + stds, color=color, alpha=0.1)

    ax.set_xscale("log", base=2)
    ax.set_xticks(args.dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title("Latent Bottleneck Capacity Sweep Across Task Dimensions", fontsize=10, pad=8)
    ax.set_xlabel("Latent Dimension $d$ (log scale)", fontsize=9)
    ax.set_ylabel("Linear Probe Mean $R^2$", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_14_bottleneck_sweep.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.14 -> {fig_path}")

    out_json = os.path.join(args.results_dir, "bottleneck_sweep_metrics.json")
    save_json(sweep_results, out_json)
    print(f"Saved sweep metrics -> {out_json}")


if __name__ == "__main__":
    main()
