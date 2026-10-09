"""05_aliasing_diagnostic.py - Temporal aliasing diagnostic on DMC observations.

The experiment compares a single-frame encoder with an encoder receiving two
immediately consecutive frames. Strict temporal stacks never cross episode
boundaries. The hypothesis is stated before measurement; the script reports the
observed result without presupposing that two-frame stacking will recover velocity.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


VARIABLE_NAMES = [
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal aliasing diagnostic.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=2000)
    parser.add_argument("--max-val-samples", type=int, default=500)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    return parser.parse_args()


def train_and_eval(
    frame_stack: int,
    seed: int,
    train_raw: dict,
    val_raw: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    set_seed(seed)

    def valid_centres(raw):
        episode_ids = raw["episode_ids"]
        if frame_stack == 1:
            return np.arange(len(episode_ids), dtype=np.int64)
        centres = []
        for index in range(frame_stack - 1, len(episode_ids)):
            start = index - frame_stack + 1
            if np.all(episode_ids[start : index + 1] == episode_ids[index]):
                centres.append(index)
        return np.asarray(centres, dtype=np.int64)

    train_candidates = valid_centres(train_raw)
    val_candidates = valid_centres(val_raw)
    n_train = min(len(train_candidates), args.max_train_samples)
    n_val = min(len(val_candidates), args.max_val_samples)
    train_indices = select_episode_stratified_indices(
        train_raw["episode_ids"], n_train, candidate_indices=train_candidates
    )
    val_indices = select_episode_stratified_indices(
        val_raw["episode_ids"], n_val, candidate_indices=val_candidates
    )

    train_ds = DMCDataset(
        frames=train_raw["frames"],
        physics_states=train_raw["physics_states"],
        episode_ids=train_raw["episode_ids"],
        frame_stack=frame_stack,
        strict_frame_stack=(frame_stack > 1),
        sample_indices=train_indices,
    )
    val_ds = DMCDataset(
        frames=val_raw["frames"],
        physics_states=val_raw["physics_states"],
        episode_ids=val_raw["episode_ids"],
        frame_stack=frame_stack,
        strict_frame_stack=(frame_stack > 1),
        sample_indices=val_indices,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    encoder = ConvEncoder(in_channels=3 * frame_stack, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(
        in_dim=args.latent_dim,
        hidden_dim=64,
        out_dim=args.latent_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for _ in range(args.epochs):
        encoder.train()
        proj_head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = pipeline(images)
            p1 = proj_head(encoder(v1))
            p2 = proj_head(encoder(v2))
            loss = nt_xent_loss(p1, p2, temperature=args.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return evaluate_linear_probe(encoder, train_loader, val_loader, device)


def main() -> None:
    args = parse_args()
    device = get_device()

    train_raw = load_dataset_npz(
        os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    )
    val_raw = load_dataset_npz(
        os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")
    )

    single_results = []
    stacked_results = []

    print(f"=== Aliasing diagnostic across seeds {args.seeds} ===")
    for seed in args.seeds:
        single = train_and_eval(1, seed, train_raw, val_raw, args, device)
        stacked = train_and_eval(2, seed, train_raw, val_raw, args, device)
        single_results.append(single["r2_per_variable"])
        stacked_results.append(stacked["r2_per_variable"])

        print("Seed", seed)
        print("  single:", dict(zip(VARIABLE_NAMES, np.round(single["r2_per_variable"], 4))))
        print("  stacked:", dict(zip(VARIABLE_NAMES, np.round(stacked["r2_per_variable"], 4))))

    single_arr = np.asarray(single_results, dtype=np.float64)
    stacked_arr = np.asarray(stacked_results, dtype=np.float64)
    delta_arr = stacked_arr - single_arr

    rows = []
    for i, name in enumerate(VARIABLE_NAMES):
        rows.append({
            "variable": name,
            "single_frame_mean": float(single_arr[:, i].mean()),
            "single_frame_std": float(single_arr[:, i].std()),
            "stacked_frame_mean": float(stacked_arr[:, i].mean()),
            "stacked_frame_std": float(stacked_arr[:, i].std()),
            "delta_mean": float(delta_arr[:, i].mean()),
            "delta_std": float(delta_arr[:, i].std()),
        })

    print("\nTable 3.3 Aliasing Diagnostic")
    for row in rows:
        print(
            f"{row['variable']:<24} "
            f"single={row['single_frame_mean']:.3f} ± {row['single_frame_std']:.3f} "
            f"stacked={row['stacked_frame_mean']:.3f} ± {row['stacked_frame_std']:.3f} "
            f"delta={row['delta_mean']:+.3f}"
        )

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "seeds": args.seeds,
            "config": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "latent_dim": args.latent_dim,
                "temperature": args.temperature,
                "max_train_samples": args.max_train_samples,
                "max_val_samples": args.max_val_samples,
                "max_shift": 3,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
                "strict_frame_stack_by_stack_size": {
                    "1": False,
                    "2": True,
                },
                "sample_selection": "episode_stratified_even_within_episode",
            },
            "per_seed_single_frame_r2": single_results,
            "per_seed_stacked_frame_r2": stacked_results,
            "results": rows,
        },
        os.path.join(args.results_dir, "table_03_03_aliasing_diagnostic.json"),
    )


if __name__ == "__main__":
    main()
