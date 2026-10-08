"""table_03_04_comparison.py - Compare representation objectives.

The table reports position and velocity decodability separately. Averaging all
state variables together can hide the distinction between configuration
information and velocity information, which is central to Chapter 3.
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
    parser = argparse.ArgumentParser(description="Compile Table 3.4.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=3500)
    parser.add_argument("--max-val-samples", type=int, default=1000)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    return parser.parse_args()


def make_loaders(train_ds, val_ds, batch_size, seed):
    generator = torch.Generator(); generator.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def summarize_probe(result: dict, state_dim: int) -> dict:
    r2 = np.asarray(result["r2_per_variable"], dtype=np.float64)
    n_position = state_dim // 2
    position = float(np.mean(r2[:n_position]))
    velocity = float(np.mean(r2[n_position:]))
    return {
        "r2_per_variable": r2.tolist(),
        "mean_r2": float(np.mean(r2)),
        "position_mean_r2": position,
        "velocity_mean_r2": velocity,
    }


def train_supervised(train_loader, val_loader, state_dim, args, device, seed):
    set_seed(seed)
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    head = nn.Linear(args.latent_dim, state_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=args.lr)
    for _ in range(args.epochs):
        encoder.train(); head.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["physics_state"].to(device)
            pred = head(encoder(x))
            loss = nn.functional.mse_loss(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return summarize_probe(evaluate_linear_probe(encoder, train_loader, val_loader, device), state_dim)


def train_contrastive(train_loader, val_loader, args, device, seed, state_dim):
    set_seed(seed)
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.lr)
    pipeline = ViewPipeline()
    for _ in range(args.epochs):
        encoder.train(); projector.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            loss = nt_xent_loss(projector(encoder(v1)), projector(encoder(v2)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return summarize_probe(evaluate_linear_probe(encoder, train_loader, val_loader, device), state_dim)


def train_vicreg(train_loader, val_loader, args, device, seed, state_dim):
    set_seed(seed)
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=args.lr)
    pipeline = ViewPipeline()
    for _ in range(args.epochs):
        encoder.train(); projector.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            loss, _, _, _ = vicreg_loss(projector(encoder(v1)), projector(encoder(v2)))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return summarize_probe(evaluate_linear_probe(encoder, train_loader, val_loader, device), state_dim)


def train_mae(train_loader, val_loader, args, device, seed, state_dim):
    set_seed(seed)
    model = SimpleMAE(latent_dim=args.latent_dim, mask_ratio=0.75).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.epochs):
        model.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            loss, _, _ = model(x)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = evaluate_linear_probe(MAEEncoder(model), train_loader, val_loader, device)
    return summarize_probe(result, state_dim)


def main() -> None:
    args = parse_args()
    device = get_device()
    task_specs = [
        ("cartpole_balance", 4),
        ("finger_spin", 6),
    ]
    methods = [
        ("Supervised State-Prediction Reference", train_supervised),
        ("Contrastive (NT-Xent)", train_contrastive),
        ("VICReg", train_vicreg),
        ("Masked Autoencoder (MAE)", train_mae),
    ]

    table = {}
    for task_name, state_dim in task_specs:
        train_raw = load_dataset_npz(os.path.join(args.data_dir, f"dmc_{task_name}_train.npz"))
        val_raw = load_dataset_npz(os.path.join(args.data_dir, f"dmc_{task_name}_val.npz"))
        n_train = min(len(train_raw["frames"]), args.max_train_samples)
        n_val = min(len(val_raw["frames"]), args.max_val_samples)
        train_ds = DMCDataset(train_raw["frames"][:n_train], train_raw["physics_states"][:n_train], train_raw["episode_ids"][:n_train])
        val_ds = DMCDataset(val_raw["frames"][:n_val], val_raw["physics_states"][:n_val], val_raw["episode_ids"][:n_val])

        table[task_name] = {}
        for name, fn in methods:
            seed_results = []
            for seed in args.seeds:
                train_loader, val_loader = make_loaders(train_ds, val_ds, args.batch_size, seed)
                result = fn(train_loader, val_loader, state_dim, args, device, seed) if name.startswith("Supervised") else fn(train_loader, val_loader, args, device, seed, state_dim)
                seed_results.append(result)
                print(
                    f"{task_name} | {name} | seed={seed} | "
                    f"position={result['position_mean_r2']:.3f} | "
                    f"velocity={result['velocity_mean_r2']:.3f}"
                )
            table[task_name][name] = {
                "position_mean": float(np.mean([r["position_mean_r2"] for r in seed_results])),
                "position_std": float(np.std([r["position_mean_r2"] for r in seed_results])),
                "velocity_mean": float(np.mean([r["velocity_mean_r2"] for r in seed_results])),
                "velocity_std": float(np.std([r["velocity_mean_r2"] for r in seed_results])),
                "mean_r2": float(np.mean([r["mean_r2"] for r in seed_results])),
                "std_r2": float(np.std([r["mean_r2"] for r in seed_results])),
                "seed_results": seed_results,
            }

    print("\nTable 3.4: position and velocity probe R^2")
    for method in [m[0] for m in methods]:
        cp = table["cartpole_balance"][method]
        fg = table["finger_spin"][method]
        print(
            f"{method:<38} | "
            f"CP pos {cp['position_mean']:.3f} ± {cp['position_std']:.3f} | "
            f"CP vel {cp['velocity_mean']:.3f} ± {cp['velocity_std']:.3f} | "
            f"FG pos {fg['position_mean']:.3f} ± {fg['position_std']:.3f} | "
            f"FG vel {fg['velocity_mean']:.3f} ± {fg['velocity_std']:.3f}"
        )

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "config": vars(args),
            "data": table,
        },
        os.path.join(args.results_dir, "table_03_04_objective_comparison.json"),
    )


if __name__ == "__main__":
    main()
