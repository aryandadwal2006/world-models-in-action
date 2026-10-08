"""10_bottleneck_sweep.py - Measure representation capacity across latent sizes.

The experiment tests, rather than assumes, whether state decodability saturates
at task-dependent latent dimensions. The Ridge probe is evaluated with a stable
float64 least-squares solver, so extreme high-dimensional results are not caused
by an avoidable normal-equation conditioning failure.
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
    parser = argparse.ArgumentParser(description="Latent-capacity sweep.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dims", type=int, nargs="+", default=[2, 4, 8, 16, 64, 256])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures")
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    return parser.parse_args()


def train_and_evaluate(train_loader, val_loader, latent_dim, seed, args, device):
    set_seed(seed)
    encoder = ConvEncoder(latent_dim=latent_dim).to(device)
    projector = ProjectionHead(latent_dim, 64, latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline()
    for _ in range(args.epochs):
        encoder.train(); projector.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            loss = nt_xent_loss(projector(encoder(v1)), projector(encoder(v2)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return evaluate_linear_probe(encoder, train_loader, val_loader, device)


def main() -> None:
    args = parse_args()
    device = get_device()
    tasks = [
        ("cartpole_balance", "Cartpole Balance (4-D state)"),
        ("cheetah_run", "Cheetah Run (18-D state)"),
    ]
    sweep_results = {}

    for task_name, task_title in tasks:
        train_raw = load_dataset_npz(os.path.join(args.data_dir, f"dmc_{task_name}_train.npz"))
        val_raw = load_dataset_npz(os.path.join(args.data_dir, f"dmc_{task_name}_val.npz"))
        n_train = min(len(train_raw["frames"]), args.max_train_samples)
        n_val = min(len(val_raw["frames"]), args.max_val_samples)
        train_ds = DMCDataset(
            train_raw["frames"][:n_train],
            train_raw["physics_states"][:n_train],
            train_raw["episode_ids"][:n_train],
        )
        val_ds = DMCDataset(
            val_raw["frames"][:n_val],
            val_raw["physics_states"][:n_val],
            val_raw["episode_ids"][:n_val],
        )

        task_data = {
            "dims": args.dims,
            "seeds": args.seeds,
            "mean_r2": [],
            "std_r2": [],
            "r2_matrix": [],
        }
        for d in args.dims:
            scores = []
            for seed in args.seeds:
                generator = torch.Generator(); generator.manual_seed(seed)
                train_loader = DataLoader(
                    train_ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    generator=generator,
                )
                val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                result = train_and_evaluate(train_loader, val_loader, d, seed, args, device)
                scores.append(float(result["mean_r2"]))
            task_data["r2_matrix"].append(scores)
            task_data["mean_r2"].append(float(np.mean(scores)))
            task_data["std_r2"].append(float(np.std(scores)))
            print(f"{task_title} d={d}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
        sweep_results[task_name] = task_data

    os.makedirs(args.figures_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    styles = {
        "cartpole_balance": ("-", "Cartpole Balance (4-D state)"),
        "cheetah_run": ("--", "Cheetah Run (18-D state)"),
    }
    for task_name, (linestyle, label) in styles.items():
        d = sweep_results[task_name]
        dims = d["dims"]
        means = np.asarray(d["mean_r2"])
        stds = np.asarray(d["std_r2"])
        ax.plot(dims, means, linestyle=linestyle, marker="o", linewidth=1.8, label=label)
        ax.fill_between(dims, means - stds, means + stds, alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Latent Dimension $d$ (log scale)")
    ax.set_ylabel("Linear Probe Mean $R^2$")
    ax.set_title("Latent Capacity Sweep Across Task Dimensions")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.figures_dir, "fig03_14_bottleneck_sweep.png"), dpi=300, bbox_inches="tight")
    plt.close()

    save_json(
        {
            "config": vars(args),
            "results": sweep_results,
        },
        os.path.join(args.results_dir, "bottleneck_sweep_metrics.json"),
    )


if __name__ == "__main__":
    main()
