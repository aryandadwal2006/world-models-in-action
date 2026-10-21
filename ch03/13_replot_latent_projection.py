"""Regenerate the manuscript's qualitative Figure 3.8 from committed data and weights.

This utility does not retrain the model. It recreates the random seed-0 encoder
and loads the committed trained seed-0 checkpoint, then plots the same
episode-stratified validation sample using a shared PCA basis.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from ch03.latent_projection import plot_latent_projection
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz, select_episode_stratified_indices
from worldmodels.eval.probes import extract_features
from worldmodels.models.encoders import ConvEncoder
from worldmodels.train import get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "contrastive_encoder_seed_0.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "ch03" / "figures" / "fig03_07_latent_projection_seed_0.png")
    parser.add_argument("--max-validation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_validation_samples < 2:
        raise ValueError("--max-validation-samples must be at least 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    data_path = args.data_dir / "dmc_cartpole_balance_val.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"Validation cache not found: {data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Trained encoder checkpoint not found: {args.checkpoint}")

    device = get_device()
    set_seed(0)
    val_raw = load_dataset_npz(str(data_path))
    n_val = min(len(val_raw["frames"]), args.max_validation_samples)
    indices = select_episode_stratified_indices(val_raw["episode_ids"], n_val)
    dataset = DMCDataset(
        frames=val_raw["frames"],
        physics_states=val_raw["physics_states"],
        episode_ids=val_raw["episode_ids"],
        sample_indices=indices,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Seed zero recreates the pre-training visual encoder used by the runner.
    encoder = ConvEncoder(in_channels=3, latent_dim=16).to(device)
    z_untrained, state_targets = extract_features(encoder, loader, device)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    stored_seed = checkpoint.get("seed")
    if stored_seed != 0:
        raise ValueError(f"Expected seed-0 checkpoint, got seed={stored_seed}")
    if int(checkpoint.get("latent_dim", -1)) != 16:
        raise ValueError("Expected a 16-dimensional encoder checkpoint")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    z_trained, _ = extract_features(encoder, loader, device)

    episode_counts = {
        str(episode): int((val_raw["episode_ids"][indices] == episode).sum())
        for episode in sorted(set(val_raw["episode_ids"][indices].tolist()))
    }
    plot_latent_projection(z_untrained, z_trained, state_targets, args.output)
    print(f"Saved shared-basis Figure 3.8 projection to {args.output}")
    print(f"Validation samples: {len(indices)}; episode counts: {episode_counts}")


if __name__ == "__main__":
    main()
