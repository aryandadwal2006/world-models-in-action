"""table_03_04_comparison.py - Collate objective comparison (Table 3.4).

Evaluates and compiles the representation quality comparison across four learning paradigms:
1. Contrastive Learning (NT-Xent)
2. Non-Contrastive Regularization (VICReg)
3. Masked Autoencoding (MAE)
4. Supervised Baseline (ConvEncoder trained directly on privileged physics states)
Evaluated across two DMC tasks: cartpole_balance and finger_spin.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import nt_xent_loss, vicreg_loss
from worldmodels.models.encoders import ConvEncoder, MAEEncoder, ProjectionHead, SimpleMAE
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile Table 3.4 objective comparison.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def train_supervised_baseline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_dim: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    """Trains ConvEncoder + Linear head directly on ground truth physics states."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    head = nn.Linear(args.latent_dim, target_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["physics_state"].to(device)

            preds = head(encoder(images))
            loss = nn.functional.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    return probe["mean_r2"]


def train_contrastive_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    """Trains ConvEncoder with NT-Xent."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)
            loss = nt_xent_loss(proj(encoder(v1)), proj(encoder(v2)), temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    return probe["mean_r2"]


def train_vicreg_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    """Trains ConvEncoder with VICReg."""
    set_seed(seed)
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=args.lr)
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)
            loss, _, _, _ = vicreg_loss(proj(encoder(v1)), proj(encoder(v2)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    return probe["mean_r2"]


def train_mae_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> float:
    """Trains SimpleMAE and evaluates linear probe."""
    set_seed(seed)
    mae = SimpleMAE(latent_dim=args.latent_dim, mask_ratio=0.75).to(device)
    optimizer = torch.optim.Adam(mae.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        mae.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            loss, _, _ = mae(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(MAEEncoder(mae), train_loader, val_loader, device)
    return probe["mean_r2"]


def main() -> None:
    args = parse_args()
    device = get_device()

    tasks = [
        ("cartpole_balance", 4),
        ("finger_spin", 6),
    ]

    table_data = {}

    for task_name, state_dim in tasks:
        print(f"\n================ Running Table 3.4 for {task_name} ================")
        train_file = os.path.join(args.data_dir, f"dmc_{task_name}_train.npz")
        val_file = os.path.join(args.data_dir, f"dmc_{task_name}_val.npz")

        train_raw = load_dataset_npz(train_file)
        val_raw = load_dataset_npz(val_file)

        n_train = min(len(train_raw["frames"]), 3500)
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

        methods = {
            "Supervised (Upper Bound)": lambda s: train_supervised_baseline(train_loader, val_loader, state_dim, args, device, s),
            "Contrastive (NT-Xent)": lambda s: train_contrastive_model(train_loader, val_loader, args, device, s),
            "VICReg": lambda s: train_vicreg_model(train_loader, val_loader, args, device, s),
            "Masked Autoencoder (MAE)": lambda s: train_mae_model(train_loader, val_loader, args, device, s),
        }

        task_scores = {}
        for method_name, fn in methods.items():
            print(f"  Evaluating {method_name} across seeds {args.seeds}...")
            scores = []
            for seed in args.seeds:
                r2 = fn(seed)
                scores.append(r2)
                print(f"    Seed {seed}: R^2 = {r2:.4f}")
            task_scores[method_name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "scores": scores,
            }
        table_data[task_name] = task_scores

    # Print Table 3.4
    print("\n" + "=" * 80)
    print("Table 3.4 Objective Comparison: Linear Probe R^2 across Two Tasks (Mean ± Std)")
    print("=" * 80)
    print(f"{'Method / Paradigm':<28} | {'Cartpole Balance (4-D)':<24} | {'Finger Spin (6-D)':<24}")
    print("-" * 80)
    method_keys = ["Supervised (Upper Bound)", "Contrastive (NT-Xent)", "VICReg", "Masked Autoencoder (MAE)"]
    for m in method_keys:
        cp_str = f"{table_data['cartpole_balance'][m]['mean']:.3f} ± {table_data['cartpole_balance'][m]['std']:.3f}"
        fg_str = f"{table_data['finger_spin'][m]['mean']:.3f} ± {table_data['finger_spin'][m]['std']:.3f}"
        print(f"{m:<28} | {cp_str:<24} | {fg_str:<24}")
    print("=" * 80)

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "table_03_04_objective_comparison.json")
    save_json({"seeds": args.seeds, "data": table_data}, out_json)
    print(f"Saved Table 3.4 -> {out_json}")


if __name__ == "__main__":
    main()
