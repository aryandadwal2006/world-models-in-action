"""07_three_mechanisms.py - Comparative verification of anti-collapse mechanisms.

Verifies three distinct non-contrastive mechanisms that prevent representation collapse
without negative pairs:
1. Stop-gradient with asymmetric prediction head (SimSiam; Eq 3.9, 3.10)
2. Exponential moving average target network (BYOL; Eq 3.11)
3. Feature cross-correlation redundancy reduction (Barlow Twins; Eq 3.12, 3.13)
"""

from __future__ import annotations

import argparse
import copy
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
from worldmodels.losses.contrastive import (
    barlow_twins_loss,
    simsiam_loss,
    update_target_ema,
)
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify three anti-collapse mechanisms.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per mechanism.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def train_simsiam(train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict:
    """Trains with SimSiam: stop-gradient on target, predictor on online branch."""
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)
    predictor = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        predictor.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)

            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))
            p1 = predictor(z1)
            p2 = predictor(z2)

            loss = simsiam_loss(p1, z2, p2, z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    # Check latent std
    encoder.eval()
    all_z = [encoder(b["image"].to(device)).detach().cpu() for b in train_loader]
    std = float(torch.cat(all_z, dim=0).std(dim=0).mean().item())
    return {"mean_r2": probe["mean_r2"], "latent_std": std}


def train_byol(train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict:
    """Trains with BYOL: online network + EMA target network + predictor."""
    online_enc = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    online_proj = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)
    predictor = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

    target_enc = copy.deepcopy(online_enc).to(device)
    target_proj = copy.deepcopy(online_proj).to(device)
    for p in target_enc.parameters():
        p.requires_grad = False
    for p in target_proj.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        list(online_enc.parameters()) + list(online_proj.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        online_enc.train()
        online_proj.train()
        predictor.train()

        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)

            # Online branch predicts target branch
            p1 = predictor(online_proj(online_enc(v1)))
            p2 = predictor(online_proj(online_enc(v2)))

            with torch.no_grad():
                z1_target = target_proj(target_enc(v1))
                z2_target = target_proj(target_enc(v2))

            loss = simsiam_loss(p1, z2_target, p2, z1_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_target_ema(online_enc, target_enc, decay=0.99)
            update_target_ema(online_proj, target_proj, decay=0.99)

    probe = evaluate_linear_probe(online_enc, train_loader, val_loader, device)
    online_enc.eval()
    all_z = [online_enc(b["image"].to(device)).detach().cpu() for b in train_loader]
    std = float(torch.cat(all_z, dim=0).std(dim=0).mean().item())
    return {"mean_r2": probe["mean_r2"], "latent_std": std}


def train_barlow_twins(train_loader: DataLoader, val_loader: DataLoader, args: argparse.Namespace, device: torch.device) -> dict:
    """Trains with Barlow Twins: cross-correlation redundancy reduction."""
    encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)

            z1 = proj_head(encoder(v1))
            z2 = proj_head(encoder(v2))

            loss = barlow_twins_loss(z1, z2, lambd=0.005)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    encoder.eval()
    all_z = [encoder(b["image"].to(device)).detach().cpu() for b in train_loader]
    std = float(torch.cat(all_z, dim=0).std(dim=0).mean().item())
    return {"mean_r2": probe["mean_r2"], "latent_std": std}


def main() -> None:
    args = parse_args()
    device = get_device()

    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    val_file = os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")

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

    mechanisms = [
        ("SimSiam (Stop-Gradient)", train_simsiam),
        ("BYOL (EMA Target)", train_byol),
        ("Barlow Twins (Cross-Corr)", train_barlow_twins),
    ]

    summary_results = {}

    print("=== Testing Three Anti-Collapse Mechanisms across Seeds ===")
    for name, train_fn in mechanisms:
        r2_scores = []
        std_scores = []
        for seed in args.seeds:
            set_seed(seed)
            print(f"Running {name} [Seed {seed}]...")
            res = train_fn(train_loader, val_loader, args, device)
            r2_scores.append(res["mean_r2"])
            std_scores.append(res["latent_std"])
            print(f"  R^2: {res['mean_r2']:.4f} | Latent Std: {res['latent_std']:.4f}")

        summary_results[name] = {
            "r2_mean": float(np.mean(r2_scores)),
            "r2_std": float(np.std(r2_scores)),
            "latent_std_mean": float(np.mean(std_scores)),
            "latent_std_std": float(np.std(std_scores)),
        }

    print("\n" + "=" * 65)
    print("Table 3.1 Verification: Three Anti-Collapse Mechanisms")
    print("=" * 65)
    print(f"{'Mechanism':<28} | {'Latent Std (Anti-Collapse)':<20} | {'Probe R^2':<12}")
    print("-" * 65)
    for name, stats in summary_results.items():
        std_str = f"{stats['latent_std_mean']:.3f} ± {stats['latent_std_std']:.3f}"
        r2_str = f"{stats['r2_mean']:.3f} ± {stats['r2_std']:.3f}"
        print(f"{name:<28} | {std_str:<20} | {r2_str:<12}")
    print("=" * 65)

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "three_mechanisms_verification.json")
    save_json(summary_results, out_json)
    print(f"Saved results -> {out_json}")


if __name__ == "__main__":
    main()
