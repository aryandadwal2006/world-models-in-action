"""07_three_mechanisms.py - Compare three anti-collapse mechanisms.

Raw latent standard deviation is reported as a diagnostic, but the comparison
metric is the scale-invariant effective rank of the representation covariance.
This avoids interpreting arbitrary latent scaling as representation quality.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.noncontrastive import (
    barlow_twins_loss,
    simsiam_loss,
    update_target_ema,
)
from worldmodels.models.encoders import ConvEncoder
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare anti-collapse mechanisms.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--projector-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-samples", type=int, default=6000)
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--brightness-range", type=float, default=0.05)
    parser.add_argument("--contrast-range", type=float, default=0.05)
    parser.add_argument("--byol-ema-start", type=float, default=0.99)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    return parser.parse_args()


class SimSiamProjector(nn.Module):
    """Reduced-width version of the paper's 3-layer projector with BatchNorm."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),
        )

    def forward(self, x):
        return self.net(x)


class SimSiamPredictor(nn.Module):
    """Two-layer predictor with hidden BatchNorm, matching SimSiam's mechanism."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class BYOLHead(nn.Module):
    """Small two-layer BYOL MLP; width is intentionally far below ImageNet scale."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class BarlowProjector(nn.Module):
    """Reduced-width 3-layer Barlow Twins projector with BatchNorm hidden layers."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def _ema_update_buffers(online_model: nn.Module, target_model: nn.Module, decay: float):
    """Keep target BatchNorm running statistics synchronized with the EMA target."""
    for online_buffer, target_buffer in zip(
        online_model.buffers(), target_model.buffers()
    ):
        if target_buffer.is_floating_point():
            target_buffer.mul_(decay).add_(online_buffer, alpha=1.0 - decay)
        else:
            target_buffer.copy_(online_buffer)


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


def _feature_statistics(features: np.ndarray) -> dict:
    """Summarize one representation space without confusing scale and rank."""
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
    dimension_std = np.std(features, axis=0)
    return {
        "effective_rank": rank,
        "mean_dim_std": float(dimension_std.mean()),
        "min_dim_std": float(dimension_std.min()),
        "max_dim_std": float(dimension_std.max()),
        "mean_feature_variance": float(np.var(features, axis=0).mean()),
        "per_dimension_std": dimension_std.tolist(),
    }


def _representation_stats(
    encoder,
    loader,
    device,
    projector=None,
    predictor=None,
    target_encoder=None,
    target_projector=None,
) -> dict:
    """Measure the encoder space and every available loss-space branch."""
    modules = [encoder, projector, predictor, target_encoder, target_projector]
    previous_modes = {
        id(module): module.training
        for module in modules
        if module is not None
    }
    for module in modules:
        if module is not None:
            module.eval()

    collected = {
        "encoder": [],
        "projector": [],
        "predictor": [],
        "target_projector": [],
    }
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            encoded = encoder(images)
            collected["encoder"].append(encoded.cpu().numpy())

            projected = None
            if projector is not None:
                projected = projector(encoded)
                collected["projector"].append(projected.cpu().numpy())
            if predictor is not None:
                if projected is None:
                    raise ValueError("predictor diagnostics require a projector")
                predicted = predictor(projected)
                collected["predictor"].append(predicted.cpu().numpy())
            if target_encoder is not None or target_projector is not None:
                if target_encoder is None or target_projector is None:
                    raise ValueError(
                        "target_encoder and target_projector must be provided together"
                    )
                target_features = target_projector(target_encoder(images))
                collected["target_projector"].append(target_features.cpu().numpy())

    diagnostics = {}
    for name, batches in collected.items():
        if batches:
            diagnostics[name] = _feature_statistics(np.concatenate(batches, axis=0))
    for module in modules:
        if module is not None:
            module.train(previous_modes[id(module)])
    return diagnostics


def _snapshot_epochs(num_epochs: int) -> set[int]:
    return {
        1,
        max(1, num_epochs // 3),
        max(1, (2 * num_epochs) // 3),
        num_epochs,
    }


def _append_training_snapshot(
    history, epoch, epoch_loss, encoder, diagnostic_loader, device,
    projector=None, predictor=None, target_encoder=None, target_projector=None,
):
    diagnostics = _representation_stats(
        encoder, diagnostic_loader, device, projector=projector,
        predictor=predictor, target_encoder=target_encoder,
        target_projector=target_projector,
    )
    history.append({
        "epoch": int(epoch),
        "mean_training_loss": float(epoch_loss),
        "representation_diagnostics": diagnostics,
    })


def _validate_args(args):
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError("epochs >= 1, batch-size >= 2, and lr > 0 are required")
    if min(args.latent_dim, args.projector_dim, args.hidden_dim) < 2:
        raise ValueError("latent, projector, and hidden dimensions must each be >= 2")
    if args.max_train_samples < 2:
        raise ValueError("max-train-samples must be >= 2")
    if args.max_shift < 0 or args.brightness_range < 0 or args.contrast_range < 0:
        raise ValueError("augmentation ranges must be non-negative")
    if not 0 <= args.byol_ema_start < 1:
        raise ValueError("byol-ema-start must be in [0, 1)")


def train_simsiam(train_loader, val_loader, args, device, diagnostic_loader):
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = SimSiamProjector(args.latent_dim, args.hidden_dim, args.projector_dim).to(device)
    predictor = SimSiamPredictor(args.projector_dim, args.hidden_dim, args.projector_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()) + list(predictor.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(
        max_shift=args.max_shift,
        brightness_range=args.brightness_range,
        contrast_range=args.contrast_range,
    )
    history = []
    snapshots = _snapshot_epochs(args.epochs)
    final_epoch_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projector.train()
        predictor.train()
        loss_total = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            z1 = projector(encoder(v1))
            z2 = projector(encoder(v2))
            p1 = predictor(z1)
            p2 = predictor(z2)
            loss = simsiam_loss(p1, z2, p2, z1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite SimSiam loss at epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("SimSiam training loader produced no batches")
        final_epoch_loss = loss_total / n_batches
        if epoch in snapshots:
            _append_training_snapshot(
                history, epoch, final_epoch_loss, encoder,
                diagnostic_loader, device, projector=projector, predictor=predictor,
            )
    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    diagnostics = _representation_stats(
        encoder, diagnostic_loader, device, projector=projector, predictor=predictor
    )
    return {
        **probe,
        "effective_rank": diagnostics["encoder"]["effective_rank"],
        "latent_std": diagnostics["encoder"]["mean_dim_std"],
        "representation_diagnostics": diagnostics,
        "final_mean_training_loss": final_epoch_loss,
        "training_history": history,
    }


def train_byol(train_loader, val_loader, args, device, diagnostic_loader):
    online_encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    online_projector = BYOLHead(args.latent_dim, args.hidden_dim, args.projector_dim).to(device)
    predictor = BYOLHead(args.projector_dim, args.hidden_dim, args.projector_dim).to(device)
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
    pipeline = ViewPipeline(
        max_shift=args.max_shift,
        brightness_range=args.brightness_range,
        contrast_range=args.contrast_range,
    )
    history = []
    snapshots = _snapshot_epochs(args.epochs)
    total_steps = max(1, args.epochs * len(train_loader))
    global_step = 0
    final_epoch_loss = float("nan")
    final_decay = float(args.byol_ema_start)
    for epoch in range(1, args.epochs + 1):
        online_encoder.train()
        online_projector.train()
        predictor.train()
        target_encoder.train()
        target_projector.train()
        loss_total = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            p1 = predictor(online_projector(online_encoder(v1)))
            p2 = predictor(online_projector(online_encoder(v2)))
            with torch.no_grad():
                z1 = target_projector(target_encoder(v1))
                z2 = target_projector(target_encoder(v2))
            loss = simsiam_loss(p1, z2, p2, z1)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite BYOL loss at epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # BYOL's target EMA coefficient increases smoothly toward 1.
            progress = min(1.0, global_step / total_steps)
            decay = 1.0 - (1.0 - args.byol_ema_start) * (
                math.cos(math.pi * progress) + 1.0
            ) / 2.0
            update_target_ema(online_encoder, target_encoder, decay=decay)
            update_target_ema(online_projector, target_projector, decay=decay)
            _ema_update_buffers(online_encoder, target_encoder, decay=decay)
            _ema_update_buffers(online_projector, target_projector, decay=decay)
            final_decay = float(decay)
            global_step += 1
            loss_total += float(loss.detach().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("BYOL training loader produced no batches")
        final_epoch_loss = loss_total / n_batches
        if epoch in snapshots:
            _append_training_snapshot(
                history, epoch, final_epoch_loss, online_encoder,
                diagnostic_loader, device,
                projector=online_projector, predictor=predictor,
                target_encoder=target_encoder, target_projector=target_projector,
            )
    probe = evaluate_linear_probe(online_encoder, train_loader, val_loader, device)
    diagnostics = _representation_stats(
        online_encoder, diagnostic_loader, device,
        projector=online_projector, predictor=predictor,
        target_encoder=target_encoder, target_projector=target_projector,
    )
    return {
        **probe,
        "effective_rank": diagnostics["encoder"]["effective_rank"],
        "latent_std": diagnostics["encoder"]["mean_dim_std"],
        "representation_diagnostics": diagnostics,
        "final_mean_training_loss": final_epoch_loss,
        "training_history": history,
        "byol_ema_start": float(args.byol_ema_start),
        "byol_ema_final": final_decay,
    }


def train_barlow(train_loader, val_loader, args, device, diagnostic_loader):
    encoder = ConvEncoder(latent_dim=args.latent_dim).to(device)
    projector = BarlowProjector(args.latent_dim, args.hidden_dim, args.projector_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(
        max_shift=args.max_shift,
        brightness_range=args.brightness_range,
        contrast_range=args.contrast_range,
    )
    history = []
    snapshots = _snapshot_epochs(args.epochs)
    final_epoch_loss = float("nan")
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        projector.train()
        loss_total = 0.0
        n_batches = 0
        for batch in train_loader:
            x = batch["image"].to(device)
            v1, v2 = pipeline(x)
            z1 = projector(encoder(v1))
            z2 = projector(encoder(v2))
            loss = barlow_twins_loss(z1, z2, lambd=0.005)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite Barlow Twins loss at epoch={epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().item())
            n_batches += 1
        if n_batches == 0:
            raise RuntimeError("Barlow Twins training loader produced no batches")
        final_epoch_loss = loss_total / n_batches
        if epoch in snapshots:
            _append_training_snapshot(
                history, epoch, final_epoch_loss, encoder,
                diagnostic_loader, device, projector=projector,
            )
    probe = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    diagnostics = _representation_stats(
        encoder, diagnostic_loader, device, projector=projector
    )
    return {
        **probe,
        "effective_rank": diagnostics["encoder"]["effective_rank"],
        "latent_std": diagnostics["encoder"]["mean_dim_std"],
        "representation_diagnostics": diagnostics,
        "final_mean_training_loss": final_epoch_loss,
        "training_history": history,
    }


def main() -> None:
    args = parse_args()
    _validate_args(args)
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
            drop_last = len(train_ds) > args.batch_size and len(train_ds) % args.batch_size == 1
            train_loader = DataLoader(
                train_ds, batch_size=args.batch_size, shuffle=True,
                generator=generator, drop_last=drop_last,
            )
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
            diagnostic_count = min(1024, len(train_ds))
            diagnostic_indices = np.linspace(
                0, len(train_ds) - 1, num=diagnostic_count, dtype=np.int64
            ).tolist()
            diagnostic_loader = DataLoader(
                Subset(train_ds, diagnostic_indices),
                batch_size=args.batch_size, shuffle=False,
            )
            result = train_fn(
                train_loader, val_loader, args, device, diagnostic_loader
            )
            seed_results.append(result)
            print(
                f"{name} seed={seed}: R2={result['mean_r2']:.3f}, "
                f"effective-rank={result['effective_rank']:.3f}"
            )
        space_names = sorted({
            space
            for result in seed_results
            for space in result["representation_diagnostics"]
        })
        space_summary = {}
        for space in space_names:
            rows = [
                result["representation_diagnostics"][space]
                for result in seed_results
                if space in result["representation_diagnostics"]
            ]
            space_summary[space] = {
                "mean_effective_rank": float(np.mean([r["effective_rank"] for r in rows])),
                "std_effective_rank": float(np.std([r["effective_rank"] for r in rows])),
                "mean_dim_std": float(np.mean([r["mean_dim_std"] for r in rows])),
                "std_dim_std": float(np.std([r["mean_dim_std"] for r in rows])),
            }
        history_epochs = sorted({
            int(item["epoch"])
            for result in seed_results
            for item in result["training_history"]
        })
        history_summary = []
        for epoch in history_epochs:
            epoch_rows = []
            for result in seed_results:
                match = next(
                    (item for item in result["training_history"] if item["epoch"] == epoch),
                    None,
                )
                if match is not None:
                    epoch_rows.append(match)
            epoch_entry = {
                "epoch": epoch,
                "mean_training_loss": float(np.mean([
                    item["mean_training_loss"] for item in epoch_rows
                ])),
                "std_training_loss": float(np.std([
                    item["mean_training_loss"] for item in epoch_rows
                ])),
                "representation_spaces": {},
            }
            history_space_names = sorted({
                space
                for item in epoch_rows
                for space in item["representation_diagnostics"]
            })
            for space in history_space_names:
                rows = [
                    item["representation_diagnostics"][space]
                    for item in epoch_rows
                    if space in item["representation_diagnostics"]
                ]
                epoch_entry["representation_spaces"][space] = {
                    "mean_effective_rank": float(np.mean([
                        row["effective_rank"] for row in rows
                    ])),
                    "std_effective_rank": float(np.std([
                        row["effective_rank"] for row in rows
                    ])),
                    "mean_dim_std": float(np.mean([
                        row["mean_dim_std"] for row in rows
                    ])),
                    "std_dim_std": float(np.std([
                        row["mean_dim_std"] for row in rows
                    ])),
                    "mean_feature_variance": float(np.mean([
                        row["mean_feature_variance"] for row in rows
                    ])),
                }
            history_summary.append(epoch_entry)

        probe_state_names = (
            "cart_position",
            "pole_angle",
            "cart_velocity",
            "pole_angular_velocity",
        )
        first_probe = seed_results[0].get("r2_per_variable", [])
        per_variable_probe_summary = {}
        for index, _ in enumerate(first_probe):
            label = (
                probe_state_names[index]
                if index < len(probe_state_names)
                else f"state_{index}"
            )
            values = [
                float(result["r2_per_variable"][index])
                for result in seed_results
            ]
            per_variable_probe_summary[label] = {
                "mean_r2": float(np.mean(values)),
                "std_r2": float(np.std(values)),
                "per_seed": values,
            }

        summary[name] = {
            "mean_r2": float(np.mean([r["mean_r2"] for r in seed_results])),
            "std_r2": float(np.std([r["mean_r2"] for r in seed_results])),
            "r2_per_variable_summary": per_variable_probe_summary,
            # Backwards-compatible fields describe encoder outputs only.
            "mean_effective_rank": float(np.mean([r["effective_rank"] for r in seed_results])),
            "std_effective_rank": float(np.std([r["effective_rank"] for r in seed_results])),
            "mean_latent_std": float(np.mean([r["latent_std"] for r in seed_results])),
            "std_latent_std": float(np.std([r["latent_std"] for r in seed_results])),
            "representation_space_summary": space_summary,
            "training_history_summary": history_summary,
            "seed_results": seed_results,
        }

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "experiment_version": 5,
            "implementation_scope": "multi-seed, reduced-scale DMC mechanism study; not an exact large-scale reproduction of the original ImageNet benchmarks",
            "interpretation_contract": {
                "anti_collapse_evidence": "Use raw feature spread and effective rank together; effective rank alone is scale invariant and cannot detect near-zero representations.",
                "representation_utility": "Use held-out per-variable state-probe R2 alongside collapse diagnostics; non-collapse alone does not prove useful representation learning.",
                "training_diagnostics": "Record loss and representation spread/rank at early, middle, and final epochs for encoder and loss-space heads.",
            },
            "architecture_contract": {
                "simsiam": "3-layer BatchNorm projector with affine-free output BatchNorm; 2-layer BatchNorm predictor; stop-gradient cosine loss",
                "byol": "online and EMA target networks with BatchNorm MLP heads and cosine loss",
                "barlow_twins": "3-layer BatchNorm projector and cross-correlation redundancy-reduction loss",
                "backbone": "shared lightweight convolutional encoder rather than the original large-scale benchmark backbone",
            },
            "seeds": args.seeds,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "optimizer": "Adam for all methods in this reduced-scale DMC study",
                "learning_rate": args.lr,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
                "projector_dim": args.projector_dim,
                "max_train_samples": args.max_train_samples,
                "sample_selection": "episode_stratified_even_within_episode",
                "augmentation": {
                    "max_shift": args.max_shift,
                    "brightness_range": args.brightness_range,
                    "contrast_range": args.contrast_range,
                },
                "byol": {
                    "ema_schedule": "cosine_ramp_from_byol_ema_start_to_1",
                    "ema_start": args.byol_ema_start,
                },
                "barlow_twins": {
                    "lambda": 0.005,
                    "batch_standard_deviation_unbiased": False,
                    "normalization_note": "Batch feature standardization is performed in the loss; this matches the objective but is not code-identical to the official output BatchNorm implementation.",
                },
            },
            "methods": summary,
        },
        os.path.join(args.results_dir, "three_mechanisms_diagnostics.json"),
    )


if __name__ == "__main__":
    main()
