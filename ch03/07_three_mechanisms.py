"""07_three_mechanisms.py - Verify three anti-collapse mechanisms.

Raw latent standard deviation is reported as a diagnostic, but the comparison
metric is the scale-invariant effective rank of the representation covariance.
This avoids interpreting arbitrary latent scaling as representation quality.
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
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import (
    barlow_twins_loss,
    simsiam_loss,
    update_target_ema,
)
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify anti-collapse mechanisms.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--max-train-samples", type=int, default=3000)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    return parser.parse_args()


def effective_rank(z: np.ndarray) -> float:
    """Return the effective rank of the centered feature matrix."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(z, compute_uv=False)
    eigenvalues = singular_values**2
    total = float(eigenvalues.sum())
    if total <= 1e-15:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15))))
    return float(np.exp(entropy))


def _representation_stats(encoder, loader, device) -> tuple[float, float]:
    encoder.eval()
    features = []
    with torch.no_grad():
        for batch in loader:
            features.append(encoder(batch["image"].to(device)).cpu().numpy())
    z = np.concatenate(features, axis=0)
    latent_std = float(np.std(z, axis=0).mean())
    return effective_rank(z), latent_std


def train_simsiam(train_loader, val_loader, args, device):
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    predictor = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline()
    for _ in range(args.epochs):
        encoder.train(); projector.train(); predictor.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            z1 = projector(encoder(v1)); z2 = projector(encoder(v2))
            p1 = predictor(z1); p2 = predictor(z2)
            loss = simsiam_loss(p1, z2, p2, z1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    rank, std = _representation_stats(encoder, train_loader, device)
    return {**probe, "effective_rank": rank, "latent_std": std}


def train_byol(train_loader, val_loader, args, device):
    online_encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    online_projector = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    predictor = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
    target_encoder = copy.deepcopy(online_encoder).to(device)
    target_projector = copy.deepcopy(online_projector).to(device)
    for parameter in list(target_encoder.parameters()) + list(target_projector.parameters()):
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(
        list(online_encoder.parameters())
        + list(online_projector.parameters())
        + list(predictor.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline()
    for _ in range(args.epochs):
        online_encoder.train(); online_projector.train(); predictor.train()
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            p1 = predictor(online_projector(online_encoder(v1)))
            p2 = predictor(online_projector(online_encoder(v2)))
            with torch.no_grad():
                z1 = target_projector(target_encoder(v1))
                z2 = target_projector(target_encoder(v2))
            loss = simsiam_loss(p1, z2, p2, z1)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            update_target_ema(online_encoder, target_encoder, decay=0.99)
            update_target_ema(online_projector, target_projector, decay=0.99)
    probe = evaluate_linear_probe(online_encoder, train_loader, val_loader, device)
    rank, std = _representation_stats(online_encoder, train_loader, device)
    return {**probe, "effective_rank": rank, "latent_std": std}


def train_barlow(train_loader, val_loader, args, device):
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = ProjectionHead(args.latent_dim, 64, args.latent_dim).to(device)
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
            z1 = projector(encoder(v1)); z2 = projector(encoder(v2))
            loss = barlow_twins_loss(z1, z2, lambd=0.005)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    rank, std = _representation_stats(encoder, train_loader, device)
    return {**probe, "effective_rank": rank, "latent_std": std}


def main() -> None:
    args = parse_args()
    device = get_device()
    train_raw = load_dataset_npz(os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz"))
    val_raw = load_dataset_npz(os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz"))
    n_train = min(len(train_raw["frames"]), args.max_train_samples)
    train_indices = select_episode_stratified_indices(
        train_raw["episode_ids"], n_train
    )
    train_ds = DMCDataset(
        train_raw["frames"],
        train_raw["physics_states"],
        train_raw["episode_ids"],
        sample_indices=train_indices,
    )
    val_ds = DMCDataset(
        val_raw["frames"], val_raw["physics_states"], val_raw["episode_ids"]
    )

    methods = [
        ("SimSiam (Stop-Gradient)", train_simsiam),
        ("BYOL (EMA Target)", train_byol),
        ("Barlow Twins (Cross-Corr)", train_barlow),
    ]
    summary = {}
    for name, train_fn in methods:
        seed_results = []
        for seed in args.seeds:
            set_seed(seed)
            generator = torch.Generator(); generator.manual_seed(seed)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            result = train_fn(train_loader, val_loader, args, device)
            seed_results.append(result)
            print(
                f"{name} seed={seed}: R2={result['mean_r2']:.3f}, "
                f"effective-rank={result['effective_rank']:.3f}"
            )
        summary[name] = {
            "mean_r2": float(np.mean([r["mean_r2"] for r in seed_results])),
            "std_r2": float(np.std([r["mean_r2"] for r in seed_results])),
            "mean_effective_rank": float(np.mean([r["effective_rank"] for r in seed_results])),
            "std_effective_rank": float(np.std([r["effective_rank"] for r in seed_results])),
            "mean_latent_std": float(np.mean([r["latent_std"] for r in seed_results])),
            "std_latent_std": float(np.std([r["latent_std"] for r in seed_results])),
            "seed_results": seed_results,
        }

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "experiment_version": 2,
            "seeds": args.seeds,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "latent_dim": args.latent_dim,
                "max_train_samples": args.max_train_samples,
                "sample_selection": "episode_stratified_even_within_episode",
                "augmentation": {
                    "max_shift": 3,
                    "brightness_range": 0.1,
                    "contrast_range": 0.1,
                },
                "barlow_twins": {
                    "lambda": 0.005,
                    "batch_standard_deviation_unbiased": false,
                },
            },
            "methods": summary,
        },
        os.path.join(args.results_dir, "three_mechanisms_verification.json"),
    )


if __name__ == "__main__":
    main()
