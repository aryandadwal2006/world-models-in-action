"""04_train_contrastive.py - Contrastive representation learning with NT-Xent on DMC frames.

Trains ConvEncoder with a 2-layer MLP projection head on cartpole_balance frames
using the view augmentation pipeline and NT-Xent loss across seeds 0, 1, 2.
Produces training loss curves (Figure 3.6), latent 2D PCA projections (Figure 3.7),
and quantitative linear probe evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.eval.probes import evaluate_linear_probe, extract_features
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, load_json, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train contrastive encoder with NT-Xent.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0, 1, or 2).")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Encoder latent dimension.")
    parser.add_argument("--temperature", type=float, default=0.5, help="NT-Xent temperature tau.")
    parser.add_argument("--max-train-samples", type=int, default=5000, help="Cap training samples for CPU speed.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data cache directory.")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints", help="Checkpoint directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures output directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results output directory.")
    return parser.parse_args()


def plot_latent_projection(
    z_untrained: np.ndarray,
    z_trained: np.ndarray,
    physics_states: np.ndarray,
    filepath: str,
) -> None:
    """Plots 2D PCA projection of latent space before vs after training (Figure 3.7).

    Marker shape represents binned pole angle; grayscale intensity represents cart position.
    No color-only encodings, fully black-and-white print safe.
    """
    # Pure numpy SVD for 2D PCA projection
    def _project_2d(z: np.ndarray) -> np.ndarray:
        z_cent = z - np.mean(z, axis=0)
        u, s, vt = np.linalg.svd(z_cent, full_matrices=False)
        return z_cent @ vt[:2].T

    p_un = _project_2d(z_untrained)
    p_tr = _project_2d(z_trained)

    # Cart position is physics_states[:, 0]
    cart_pos = physics_states[:, 0]
    # Normalize cart position to [0.2, 0.8] for grayscale shading (0 = black, 1 = white)
    c_norm = (cart_pos - cart_pos.min()) / (cart_pos.max() - cart_pos.min() + 1e-8)
    gray_shades = [str(round(0.15 + 0.7 * c, 2)) for c in c_norm]

    # Pole angle is physics_states[:, 1]
    pole_angle = physics_states[:, 1]
    # Bin pole angle into 3 categories: tilted left (< -0.1), upright ([-0.1, 0.1]), tilted right (> 0.1)
    markers = np.where(pole_angle < -0.1, "v", np.where(pole_angle > 0.1, "^", "o"))

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))

    for ax, proj, title in zip(axes, [p_un, p_tr], ["Before Training (Random Weights)", "After NT-Xent Contrastive Training"]):
        # Plot each marker category
        for m, label in [("v", "Pole Left (< -0.1 rad)"), ("o", "Pole Upright ([-0.1, 0.1])"), ("^", "Pole Right (> 0.1 rad)")]:
            idx = np.where(markers == m)[0]
            if len(idx) == 0:
                continue
            # Sample subset for uncluttered scatter
            sub_idx = idx[:150]
            for i in sub_idx:
                ax.scatter(
                    proj[i, 0],
                    proj[i, 1],
                    marker=m,
                    color=gray_shades[i],
                    edgecolors="black",
                    linewidths=0.5,
                    s=32,
                    alpha=0.85,
                )
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("Principal Component 1", fontsize=9)
        ax.set_ylabel("Principal Component 2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    val_file = os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(
            f"Cached DMC data not found at {train_file}. Run ch03/01_collect_dmc_data.py first."
        )

    train_raw = load_dataset_npz(train_file)
    val_raw = load_dataset_npz(val_file)

    # Cap dataset for CPU training if requested
    n_train = min(len(train_raw["frames"]), args.max_train_samples)
    n_val = min(len(val_raw["frames"]), 1000)

    train_dataset = DMCDataset(
        frames=train_raw["frames"][:n_train],
        physics_states=train_raw["physics_states"][:n_train],
        episode_ids=train_raw["episode_ids"][:n_train],
    )
    val_dataset = DMCDataset(
        frames=val_raw["frames"][:n_val],
        physics_states=val_raw["physics_states"][:n_val],
        episode_ids=val_raw["episode_ids"][:n_val],
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

    # Extract initial representations for Figure 3.7 before training
    z_init, y_val_phys = extract_features(encoder, val_loader, device)

    # Optimizer and augmentations
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    view_pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    epoch_losses = []

    print(f"Starting contrastive training (seed={args.seed}, epochs={args.epochs}, device={device})...")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        total_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            # Create two augmented views
            v1, v2 = view_pipeline(images)

            z1 = encoder(v1)
            z2 = encoder(v2)
            p1 = proj_head(z1)
            p2 = proj_head(z2)

            loss = nt_xent_loss(p1, p2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(images)

        avg_loss = total_loss / len(train_dataset)
        epoch_losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:02d}/{args.epochs:02d} | NT-Xent Loss: {avg_loss:.4f}")

    # Evaluate linear probe on frozen trained encoder
    probe_results = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    print(f"Linear Probe Mean R^2: {probe_results['mean_r2']:.4f}")
    print(f"R^2 per variable (cart_pos, pole_angle, cart_vel, ang_vel): {probe_results['r2_per_variable']}")

    # Extract representations after training for Figure 3.7
    z_trained, _ = extract_features(encoder, val_loader, device)

    # Save outputs
    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    ckpt_path = os.path.join(args.checkpoints_dir, f"contrastive_encoder_seed_{args.seed}.pt")
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "latent_dim": args.latent_dim,
            "seed": args.seed,
        },
        ckpt_path,
    )

    metrics_path = os.path.join(args.results_dir, f"contrastive_metrics_seed_{args.seed}.json")
    save_json(
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "losses": epoch_losses,
            "final_loss": epoch_losses[-1],
            "probe_mean_r2": probe_results["mean_r2"],
            "probe_r2_per_variable": probe_results["r2_per_variable"],
        },
        metrics_path,
    )

    # Save Figure 3.7
    fig07_path = os.path.join(args.figures_dir, f"fig03_07_latent_projection_seed_{args.seed}.png")
    plot_latent_projection(z_init, z_trained, y_val_phys, fig07_path)
    print(f"Saved Figure 3.7 -> {fig07_path}")

    # Check if seeds 0, 1, 2 metrics exist to plot Figure 3.6 (3-seed loss band)
    seed_files = [os.path.join(args.results_dir, f"contrastive_metrics_seed_{s}.json") for s in [0, 1, 2]]
    if all(os.path.exists(f) for f in seed_files):
        all_losses = []
        for f in seed_files:
            m = load_json(f)
            all_losses.append(m["losses"])
        loss_mat = np.array(all_losses)  # (3, epochs)
        mean_l = np.mean(loss_mat, axis=0)
        std_l = np.std(loss_mat, axis=0)
        ep_axis = np.arange(1, len(mean_l) + 1)

        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        ax.plot(ep_axis, mean_l, color="black", linestyle="-", linewidth=2.0, label="NT-Xent Loss (Mean)")
        ax.fill_between(ep_axis, mean_l - std_l, mean_l + std_l, color="black", alpha=0.15, label="±1 Std (3 Seeds)")
        ax.set_title("Contrastive Pre-training Loss on DMC Cartpole Frames", fontsize=10, pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("NT-Xent Loss", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True, fontsize=9)
        plt.tight_layout()
        fig06_path = os.path.join(args.figures_dir, "fig03_06_contrastive_loss.png")
        plt.savefig(fig06_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved Figure 3.6 (3-seed band) -> {fig06_path}")


if __name__ == "__main__":
    main()
