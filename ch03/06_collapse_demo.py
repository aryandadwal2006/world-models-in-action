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
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.losses.ntxent import nt_xent_loss
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


def _feature_diagnostics(features: np.ndarray) -> dict:
    """Report spread and effective rank for a collected feature matrix."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must be a non-empty two-dimensional array")
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = singular_values**2
    total = float(eigenvalues.sum())
    if total <= 1e-15:
        rank = 0.0
    else:
        probabilities = eigenvalues / total
        entropy = -float(
            np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)))
        )
        rank = float(np.exp(entropy))
    return {
        "mean_dim_std": float(np.std(features, axis=0).mean()),
        "effective_rank": rank,
    }


def run_experiment(
    mode: str,
    seed: int,
    train_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Train attraction-only or NT-Xent and measure both representation spaces."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(
        in_dim=args.latent_dim,
        hidden_dim=64,
        out_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    view_pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    history = {
        "encoder_mean_std": [],
        "projector_mean_std": [],
        "encoder_effective_rank": [],
        "projector_effective_rank": [],
        "training_loss": [],
    }

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = view_pipeline(images)
            z1 = encoder(v1)
            z2 = encoder(v2)
            p1 = proj_head(z1)
            p2 = proj_head(z2)

            if mode == "attraction_only":
                # Deliberately naive objective: agreement without an anti-collapse term.
                loss = F.mse_loss(p1, p2)
            else:
                loss = nt_xent_loss(p1, p2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(images)
            total_examples += len(images)

        if total_examples == 0:
            raise RuntimeError("Training loader produced no examples")
        history["training_loss"].append(total_loss / total_examples)

        encoder.eval()
        proj_head.eval()
        all_encoder = []
        all_projector = []
        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(device)
                features = encoder(images)
                projected = proj_head(features)
                all_encoder.append(features.cpu().numpy())
                all_projector.append(projected.cpu().numpy())

        encoder_stats = _feature_diagnostics(np.concatenate(all_encoder, axis=0))
        projector_stats = _feature_diagnostics(np.concatenate(all_projector, axis=0))
        history["encoder_mean_std"].append(encoder_stats["mean_dim_std"])
        history["projector_mean_std"].append(projector_stats["mean_dim_std"])
        history["encoder_effective_rank"].append(encoder_stats["effective_rank"])
        history["projector_effective_rank"].append(projector_stats["effective_rank"])

        print(
            f"{mode} | seed={seed} | epoch={epoch:02d} | "
            f"loss={history['training_loss'][-1]:.5f} | "
            f"encoder std={encoder_stats['mean_dim_std']:.6f} | "
            f"projector std={projector_stats['mean_dim_std']:.6f}"
        )

    return history

def main() -> None:
    args = parse_args()
    device = get_device()

    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    train_raw = load_dataset_npz(train_file)
    n_train = min(len(train_raw["frames"]), 3000)
    train_indices = select_episode_stratified_indices(
        train_raw["episode_ids"], n_train
    )

    train_ds = DMCDataset(
        frames=train_raw["frames"],
        physics_states=train_raw["physics_states"],
        episode_ids=train_raw["episode_ids"],
        sample_indices=train_indices,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    collapse_runs = []
    contrastive_runs = []

    print(f"=== Running Collapse Demonstration across seeds {args.seeds} ===")
    for seed in args.seeds:
        print(f"Seed {seed} | Running attraction-only training (no negatives)...")
        stds_col = run_experiment("attraction_only", seed, train_loader, args, device)
        collapse_runs.append(stds_col)
        print(
            f"  Final encoder std: {stds_col['encoder_mean_std'][-1]:.6f} | "
            f"projector std: {stds_col['projector_mean_std'][-1]:.6f} | "
            f"attraction loss: {stds_col['training_loss'][-1]:.6f}"
        )

        print(f"Seed {seed} | Running contrastive training (with negatives)...")
        stds_con = run_experiment("contrastive", seed, train_loader, args, device)
        contrastive_runs.append(stds_con)
        print(
            f"  Final encoder std: {stds_con['encoder_mean_std'][-1]:.6f} | "
            f"projector std: {stds_con['projector_mean_std'][-1]:.6f} | "
            f"NT-Xent loss: {stds_con['training_loss'][-1]:.6f}"
        )

    col_arr = np.asarray(
        [run["encoder_mean_std"] for run in collapse_runs],
        dtype=np.float64,
    )
    con_arr = np.asarray(
        [run["encoder_mean_std"] for run in contrastive_runs],
        dtype=np.float64,
    )
    col_proj = np.asarray(
        [run["projector_mean_std"] for run in collapse_runs],
        dtype=np.float64,
    )
    con_proj = np.asarray(
        [run["projector_mean_std"] for run in contrastive_runs],
        dtype=np.float64,
    )

    epochs = np.arange(1, args.epochs + 1)

    # Figure 3.9 keeps its established encoder-space diagnostic.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        epochs, np.mean(con_arr, axis=0),
        color="black", linestyle="-", linewidth=2.0,
        label="Encoder output: NT-Xent",
    )
    ax.fill_between(
        epochs,
        np.mean(con_arr, axis=0) - np.std(con_arr, axis=0),
        np.mean(con_arr, axis=0) + np.std(con_arr, axis=0),
        color="black", alpha=0.15,
    )
    ax.plot(
        epochs, np.mean(col_arr, axis=0),
        color="black", linestyle="--", linewidth=2.0,
        label="Encoder output: attraction only",
    )
    ax.fill_between(
        epochs,
        np.mean(col_arr, axis=0) - np.std(col_arr, axis=0),
        np.mean(col_arr, axis=0) + np.std(col_arr, axis=0),
        color="black", alpha=0.1,
    )
    ax.set_title("Representation Variability During Training", fontsize=10, pad=8)
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

    # Supplemental diagnostic: inspect the actual space used by each objective.
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        epochs, np.mean(con_proj, axis=0),
        color="black", linestyle="-", linewidth=2.0,
        label="Projector output: NT-Xent",
    )
    ax.fill_between(
        epochs,
        np.mean(con_proj, axis=0) - np.std(con_proj, axis=0),
        np.mean(con_proj, axis=0) + np.std(con_proj, axis=0),
        color="black", alpha=0.15,
    )
    ax.plot(
        epochs, np.mean(col_proj, axis=0),
        color="black", linestyle="--", linewidth=2.0,
        label="Projector output: attraction only",
    )
    ax.fill_between(
        epochs,
        np.mean(col_proj, axis=0) - np.std(col_proj, axis=0),
        np.mean(col_proj, axis=0) + np.std(col_proj, axis=0),
        color="black", alpha=0.1,
    )
    ax.set_title("Variability in the Objective Space", fontsize=10, pad=8)
    ax.set_xlabel("Training Epoch", fontsize=9)
    ax.set_ylabel("Mean Per-Dimension Standard Deviation", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    objective_fig_path = os.path.join(
        args.figures_dir, "fig03_09_objective_space.png"
    )
    plt.savefig(objective_fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved objective-space diagnostic -> {objective_fig_path}")

    def aggregate_history(mode_runs, key):
        values = np.asarray(
            [run[key] for run in mode_runs],
            dtype=np.float64,
        )
        return [float(value) for value in values.mean(axis=0)]

    res_path = os.path.join(args.results_dir, "collapse_demo.json")
    save_json({
        "experiment_version": 2,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "sample_selection": "episode_stratified_even_within_episode",
        "metric_spaces": {
            "legacy_curve_and_collapse_mean_std": "encoder output",
            "projector_curve": "projector output where the objective is applied",
        },
        # Backwards-compatible keys remain encoder-output spread.
        "collapse_mean_std": aggregate_history(collapse_runs, "encoder_mean_std"),
        "contrastive_mean_std": aggregate_history(contrastive_runs, "encoder_mean_std"),
        "collapse_projector_mean_std": aggregate_history(collapse_runs, "projector_mean_std"),
        "contrastive_projector_mean_std": aggregate_history(contrastive_runs, "projector_mean_std"),
        "collapse_encoder_effective_rank": aggregate_history(collapse_runs, "encoder_effective_rank"),
        "contrastive_encoder_effective_rank": aggregate_history(contrastive_runs, "encoder_effective_rank"),
        "collapse_projector_effective_rank": aggregate_history(collapse_runs, "projector_effective_rank"),
        "contrastive_projector_effective_rank": aggregate_history(contrastive_runs, "projector_effective_rank"),
        "collapse_training_loss": aggregate_history(collapse_runs, "training_loss"),
        "contrastive_training_loss": aggregate_history(contrastive_runs, "training_loss"),
        "per_seed": {
            "attraction_only": collapse_runs,
            "contrastive": contrastive_runs,
        },
    }, res_path)
    print(f"Saved collapse diagnostics -> {res_path}")


if __name__ == "__main__":
    main()
