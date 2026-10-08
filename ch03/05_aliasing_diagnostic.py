"""05_aliasing_diagnostic.py - Temporal aliasing diagnostic on DMC observations.

Empirically tests Chapter 2's aliasing contract: single static frames carry sufficient
information to linearly decode position variables, but lack velocity information.
Stacking two consecutive frames provides the minimal temporal context required to recover
velocity, resolving the representation bottleneck (Table 3.3).
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
from worldmodels.data.dmc_data import DMCDataset, load_dataset_npz
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal aliasing diagnostic.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="List of seeds to evaluate.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs per encoder.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    return parser.parse_args()


def train_and_eval(
    frame_stack: int,
    seed: int,
    train_raw: dict,
    val_raw: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Trains a contrastive encoder on either single or stacked frames and evaluates linear probe."""
    import gc
    gc.collect()
    set_seed(seed)

    n_train = min(len(train_raw["frames"]), 2000)
    n_val = min(len(val_raw["frames"]), 500)

    train_ds = DMCDataset(
        frames=train_raw["frames"][:n_train],
        physics_states=train_raw["physics_states"][:n_train],
        episode_ids=train_raw["episode_ids"][:n_train],
        frame_stack=frame_stack,
    )
    val_ds = DMCDataset(
        frames=val_raw["frames"][:n_val],
        physics_states=val_raw["physics_states"][:n_val],
        episode_ids=val_raw["episode_ids"][:n_val],
        frame_stack=frame_stack,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    in_channels = 3 * frame_stack
    encoder = ConvEncoder(in_channels=in_channels, latent_dim=args.latent_dim).to(device)
    proj_head = ProjectionHead(in_dim=args.latent_dim, hidden_dim=64, out_dim=args.latent_dim).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(proj_head.parameters()),
        lr=args.lr,
    )
    view_pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()
        for batch in train_loader:
            images = batch["image"].to(device)
            v1, v2 = view_pipeline(images)
            p1 = proj_head(encoder(v1))
            p2 = proj_head(encoder(v2))
            loss = nt_xent_loss(p1, p2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe_res = evaluate_linear_probe(encoder, train_loader, val_loader, device)
    return probe_res


def main() -> None:
    args = parse_args()
    device = get_device()

    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    val_file = os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")

    train_raw = load_dataset_npz(train_file)
    val_raw = load_dataset_npz(val_file)

    var_names = ["cart_position", "pole_angle", "cart_velocity", "pole_angular_velocity"]

    single_frame_results = []
    stacked_frame_results = []

    print(f"=== Running Aliasing Diagnostic across seeds {args.seeds} ===")
    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        print("Training Single-Frame (1-frame) Encoder...")
        res_1 = train_and_eval(1, seed, train_raw, val_raw, args, device)
        single_frame_results.append(res_1["r2_per_variable"])
        print(f"  1-Frame Probe R^2: {dict(zip(var_names, [round(x, 4) for x in res_1['r2_per_variable']]))}")

        print("Training Stacked-Frame (2-frame) Encoder...")
        res_2 = train_and_eval(2, seed, train_raw, val_raw, args, device)
        stacked_frame_results.append(res_2["r2_per_variable"])
        print(f"  2-Frame Probe R^2: {dict(zip(var_names, [round(x, 4) for x in res_2['r2_per_variable']]))}")

    single_arr = np.array(single_frame_results)  # (num_seeds, 4)
    stacked_arr = np.array(stacked_frame_results)  # (num_seeds, 4)

    mean_single = np.mean(single_arr, axis=0)
    std_single = np.std(single_arr, axis=0)

    mean_stacked = np.mean(stacked_arr, axis=0)
    std_stacked = np.std(stacked_arr, axis=0)

    print("\n" + "=" * 65)
    print("Table 3.3 Aliasing Diagnostic: Probe R^2 (Mean ± Std over 3 seeds)")
    print("=" * 65)
    print(f"{'State Variable':<25} | {'Single-Frame (1-frame)':<18} | {'Stacked-Frame (2-frame)':<18}")
    print("-" * 65)
    table_rows = []
    for i, var in enumerate(var_names):
        s1 = f"{mean_single[i]:.3f} ± {std_single[i]:.3f}"
        s2 = f"{mean_stacked[i]:.3f} ± {std_stacked[i]:.3f}"
        print(f"{var:<25} | {s1:<18} | {s2:<18}")
        table_rows.append({
            "variable": var,
            "single_frame_mean": float(mean_single[i]),
            "single_frame_std": float(std_single[i]),
            "stacked_frame_mean": float(mean_stacked[i]),
            "stacked_frame_std": float(std_stacked[i]),
        })
    print("=" * 65)

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "table_03_03_aliasing_diagnostic.json")
    save_json({"seeds": args.seeds, "results": table_rows}, out_json)
    print(f"Saved Table 3.3 results -> {out_json}")


if __name__ == "__main__":
    main()
