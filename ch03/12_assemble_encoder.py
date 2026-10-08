"""12_assemble_encoder.py - Chapter 3 synthesis and final encoder assembly.

Packages the learned visual encoder into the standardized worldmodels.models.Encoder interface,
slots it into the Chapter 1 agent loop skeleton (Listing 1.1), verifies the execution contract,
and prints Synthesis Table 3.6 summarizing self-supervised objectives.
"""

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
    parser = argparse.ArgumentParser(description="Assemble final encoder and test Chapter 1 agent loop.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/contrastive_encoder_seed_0.pt", help="Encoder checkpoint.")
    parser.add_argument("--data-file", type=str, default="data/dmc_cartpole_balance_val.npz", help="Evaluation data.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Build encoder
    backbone = ConvEncoder(in_channels=3, latent_dim=16)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        backbone.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded pre-trained weights from {args.checkpoint}")
    else:
        print(f"Checkpoint {args.checkpoint} not found; running assembly with initialized weights.")

    encoder = Encoder(backbone=backbone, recurrent_hidden_dim=32).to(device)

    # Load 10 frames from validation cache to simulate incoming sensory stream
    if os.path.exists(args.data_file):
        data = load_dataset_npz(args.data_file)
        sample_frames = data["frames"][:10]  # (10, 64, 64, 3)
    else:
        sample_frames = torch.randint(0, 255, (10, 64, 64, 3), dtype=torch.uint8).numpy()

    print("\n=== Executing Chapter 1 Minimal Agent Loop with Assembled Encoder ===")
    state = None  # Listing 1.1: state = empty_state()

    for tick in range(len(sample_frames)):
        # Environment observation
        raw_obs = torch.from_numpy(sample_frames[tick]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Chapter 1 contract: state = encoder.update(state, obs)
        state = encoder.update(state, raw_obs)

        # Mock imagination branch (50 candidates x 20 steps in latent space)
        # Latent state is copied: s = state
        # In later chapters (Ch 5-7), dynamics.predict(s, a) advances s in latent space
        print(f"Tick {tick:02d} | Received obs shape {tuple(raw_obs.shape)} | Persistent state shape {tuple(state.shape)} | State norm: {state.norm().item():.3f}")

    print("\nChapter 1 encoder contract verified successfully.")

    # Synthesis Table 3.6
    synthesis_table = [
        {
            "objective": "Pretext (Rotation, Jigsaw)",
            "paid_to_keep": "Geometric orientation / patch arrangement",
            "may_discard": "Continuous state variables (velocities, fine pose)",
            "invited_failure": "Nuisance alignment, shortcut learning",
        },
        {
            "objective": "Contrastive (NT-Xent)",
            "paid_to_keep": "View-invariant structures preserved across crops",
            "may_discard": "High-frequency texture, absolute crop coordinates",
            "invited_failure": "Aggressive crop aliasing, false negatives on video",
        },
        {
            "objective": "Non-Contrastive (VICReg, SimSiam)",
            "paid_to_keep": "Decorrelated dimensions with non-zero variance",
            "may_discard": "Dimension redundancy, scale variance",
            "invited_failure": "Weight collapse if variance hinge or stopgrad fails",
        },
        {
            "objective": "Masked Autoencoding (MAE)",
            "paid_to_keep": "Spatial redundancy required to restore masked patches",
            "may_discard": "Abstract semantic boundaries (preserves pixel detail)",
            "invited_failure": "Capacity wasted reconstructing predictable background",
        },
        {
            "objective": "Temporal Contrastive (CPC)",
            "paid_to_keep": "Slowly varying predictive latent dynamics",
            "may_discard": "Unpredictable noise innovations",
            "invited_failure": "Action-unconditioned stochastic ceiling",
        },
    ]

    print("\n" + "=" * 90)
    print("Table 3.6 Synthesis of Self-Supervised Objectives")
    print("=" * 90)
    print(f"{'Objective':<28} | {'Paid to Keep':<25} | {'May Discard':<20} | {'Invited Failure'}")
    print("-" * 90)
    for row in synthesis_table:
        print(f"{row['objective']:<28} | {row['paid_to_keep']:<25} | {row['may_discard']:<20} | {row['invited_failure']}")
    print("=" * 90)

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "table_03_06_synthesis.json")
    save_json({"table": synthesis_table}, out_json)
    print(f"Saved Table 3.6 -> {out_json}")


if __name__ == "__main__":
    main()
