"""12_assemble_encoder.py - Install the learned encoder into the Chapter 1 loop."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from worldmodels.data.dmc_data import load_dataset_npz
from worldmodels.models.encoders import ConvEncoder, Encoder
from worldmodels.train import get_device, save_json, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble and verify the Chapter 3 encoder.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--recurrent-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/contrastive_encoder_seed_0.pt",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/dmc_cartpole_balance_val.npz",
    )
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--results-dir", type=str, default="ch03/results")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. "
            "Run ch03/04_train_contrastive.py first."
        )
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Evaluation data not found: {args.data_file}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_latent_dim = int(checkpoint.get("latent_dim", args.latent_dim))
    if checkpoint_latent_dim != args.latent_dim:
        raise ValueError(
            f"Checkpoint latent_dim={checkpoint_latent_dim} does not match "
            f"requested latent_dim={args.latent_dim}"
        )

    backbone = ConvEncoder(in_channels=3, latent_dim=args.latent_dim)
    backbone.load_state_dict(checkpoint["encoder_state_dict"])
    encoder = Encoder(
        backbone=backbone,
        recurrent_hidden_dim=args.recurrent_hidden_dim,
    ).to(device)
    encoder.eval()

    data = load_dataset_npz(args.data_file)
    frames = data["frames"][: args.num_frames]
    if len(frames) == 0:
        raise ValueError("Evaluation cache contains no frames")

    state = None
    state_shapes = []
    with torch.no_grad():
        for frame in frames:
            observation = (
                torch.from_numpy(frame)
                .permute(2, 0, 1)
                .float()
                .unsqueeze(0)
                .to(device)
                / 255.0
            )
            state = encoder.update(state, observation)
            state_shapes.append(tuple(state.shape))

    expected_shape = (1, args.recurrent_hidden_dim)
    if any(shape != expected_shape for shape in state_shapes):
        raise RuntimeError("Encoder recurrent-state contract failed")

    print("Chapter 1 encoder contract verified successfully.")
    print(f"Observation shape: (1, 3, 64, 64)")
    print(f"Persistent state shape: {expected_shape}")

    synthesis_table = [
        {
            "objective": "Pretext",
            "paid_to_keep": "Task-defined structure",
            "may_discard": "Variables irrelevant to the pretext",
            "invited_failure": "Shortcut features",
        },
        {
            "objective": "Contrastive (NT-Xent)",
            "paid_to_keep": "Structures preserved across views",
            "may_discard": "Information removed by augmentation",
            "invited_failure": "False negatives and aggressive invariance",
        },
        {
            "objective": "Non-Contrastive",
            "paid_to_keep": "Invariant, non-collapsed coordinates",
            "may_discard": "Redundant coordinates",
            "invited_failure": "Collapse if anti-collapse mechanism fails",
        },
        {
            "objective": "Masked Autoencoding",
            "paid_to_keep": "Information useful for reconstruction",
            "may_discard": "Details irrelevant to pixel recovery",
            "invited_failure": "Capacity spent on predictable pixels",
        },
        {
            "objective": "Temporal CPC",
            "paid_to_keep": "Information useful for future prediction",
            "may_discard": "Unpredictable innovations",
            "invited_failure": "Action-unconditioned prediction ceiling",
        },
    ]

    os.makedirs(args.results_dir, exist_ok=True)
    save_json(
        {
            "encoder_contract": {
                "checkpoint": args.checkpoint,
                "latent_dim": args.latent_dim,
                "recurrent_hidden_dim": args.recurrent_hidden_dim,
                "num_frames_tested": len(frames),
                "persistent_state_shape": list(expected_shape),
            },
            "table": synthesis_table,
        },
        os.path.join(args.results_dir, "table_03_06_synthesis.json"),
    )


if __name__ == "__main__":
    main()
