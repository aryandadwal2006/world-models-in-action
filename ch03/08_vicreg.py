"""08_vicreg.py - Variance-Invariance-Covariance Regularization (VICReg) training.

Trains ConvEncoder on DMC cartpole_balance frames using the three explicit non-contrastive forces:
1. Invariance: Mean squared distance between positive views (Eq 3.15)
2. Variance: Hinge loss maintaining feature variance above gamma (Eq 3.14)
3. Covariance: Decorrelation penalty on off-diagonal feature covariance (Eq 3.16)
Evaluates linear probe performance and logs metrics for Table 3.4.
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
from worldmodels.losses.contrastive import vicreg_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvEncoder with VICReg.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dim.")
    parser.add_argument("--sim-coeff", type=float, default=25.0, help="Invariance weight.")
    parser.add_argument("--std-coeff", type=float, default=25.0, help="Variance weight.")
    parser.add_argument("--cov-coeff", type=float, default=1.0, help="Covariance weight.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    seed_results = []
    loss_breakdowns = []

    print(f"=== Training VICReg across seeds {args.seeds} ===")
    for seed in args.seeds:
        set_seed(seed)
        encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
        proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(proj_head.parameters()),
            lr=args.lr,
        )

        epoch_loss_hist = {"total": [], "sim": [], "var": [], "cov": []}

        for epoch in range(1, args.epochs + 1):
            encoder.train()
            proj_head.train()
            t_loss = s_loss = v_loss = c_loss = 0.0

            for batch in train_loader:
                images = batch["image"].to(device)
                v1, v2 = pipeline(images)

                z1 = proj_head(encoder(v1))
                z2 = proj_head(encoder(v2))

                loss, sim, var, cov = vicreg_loss(
                    z1,
                    z2,
                    sim_coeff=args.sim_coeff,
                    std_coeff=args.std_coeff,
                    cov_coeff=args.cov_coeff,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bs = len(images)
                t_loss += loss.item() * bs
                s_loss += sim.item() * bs
                v_loss += var.item() * bs
                c_loss += cov.item() * bs

            n = len(train_ds)
            epoch_loss_hist["total"].append(t_loss / n)
            epoch_loss_hist["sim"].append(s_loss / n)
            epoch_loss_hist["var"].append(v_loss / n)
            epoch_loss_hist["cov"].append(c_loss / n)

        probe_res = evaluate_linear_probe(encoder, train_loader, val_loader, device)
        seed_results.append(probe_res)
        loss_breakdowns.append(epoch_loss_hist)
        print(f"Seed {seed} | Linear Probe Mean R^2: {probe_res['mean_r2']:.4f}")

    mean_r2 = float(np.mean([r["mean_r2"] for r in seed_results]))
    std_r2 = float(np.std([r["mean_r2"] for r in seed_results]))
    print(f"\nVICReg Aggregate Probe R^2 (3 seeds): {mean_r2:.3f} ± {std_r2:.3f}")

    # Plot Figure 3.11: Convergence of the Three Forces
    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    epochs = np.arange(1, args.epochs + 1)
    avg_sim = np.mean([hist["sim"] for hist in loss_breakdowns], axis=0)
    avg_var = np.mean([hist["var"] for hist in loss_breakdowns], axis=0)
    avg_cov = np.mean([hist["cov"] for hist in loss_breakdowns], axis=0)
    avg_tot = np.mean([hist["total"] for hist in loss_breakdowns], axis=0)

    ax1.plot(epochs, avg_tot, color="black", linestyle="-", label="Total Loss")
    ax1.set_title("Total VICReg Loss", fontsize=10, pad=6)
    ax1.set_xlabel("Epoch", fontsize=9)
    ax1.set_ylabel("Loss", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2.plot(epochs, avg_sim, color="black", linestyle="-", label="Invariance (MSE)")
    ax2.plot(epochs, avg_var, color="black", linestyle="--", label="Variance Hinge")
    ax2.plot(epochs, avg_cov, color="black", linestyle=":", label="Covariance Decorr")
    ax2.set_title("Decomposed Forces", fontsize=10, pad=6)
    ax2.set_xlabel("Epoch", fontsize=9)
    ax2.set_ylabel("Component Value", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_11_vicreg_forces.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.11 -> {fig_path}")

    out_json = os.path.join(args.results_dir, "vicreg_metrics.json")
    save_json({
        "seeds": args.seeds,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "seed_results": seed_results,
    }, out_json)
    print(f"Saved VICReg results -> {out_json}")


if __name__ == "__main__":
    main()
