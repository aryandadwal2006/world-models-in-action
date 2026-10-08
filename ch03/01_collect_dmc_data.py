"""01_collect_dmc_data.py - Collect and cache Chapter 3 DMC data.

Train and validation splits are collected from independent environment seeds,
so no episode can span the split boundary. Physics targets are qpos followed by
qvel and are retained strictly for evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt

from worldmodels.data.dmc_data import collect_task_dataset, save_dataset_npz
from worldmodels.train import set_seed


TASKS = [
    ("cartpole", "balance", "cartpole_balance", 4),
    ("finger", "spin", "finger_spin", 6),
    ("cheetah", "run", "cheetah_run", 18),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and cache DMC datasets.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-train-frames", type=int, default=30000)
    parser.add_argument("--num-val-frames", type=int, default=5000)
    parser.add_argument("--val-seed-offset", type=int, default=1000003)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_train_frames <= 0 or args.num_val_frames <= 0:
        raise ValueError("Train and validation frame counts must be positive")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    sample_frames = {}

    for domain, task, task_id, expected_state_dim in TASKS:
        train_seed = args.seed
        val_seed = args.seed + args.val_seed_offset

        print(f"Collecting {task_id} train split with seed={train_seed}...")
        train_data = collect_task_dataset(
            domain_name=domain,
            task_name=task,
            num_frames=args.num_train_frames,
            seed=train_seed,
            image_size=(64, 64),
            camera_id=args.camera_id,
            split="train",
        )

        print(f"Collecting {task_id} validation split with seed={val_seed}...")
        val_data = collect_task_dataset(
            domain_name=domain,
            task_name=task,
            num_frames=args.num_val_frames,
            seed=val_seed,
            image_size=(64, 64),
            camera_id=args.camera_id,
            split="validation",
        )

        observed_dim = int(train_data["physics_states"].shape[1])
        if observed_dim != expected_state_dim:
            raise RuntimeError(
                f"{task_id}: expected qpos+qvel state dim {expected_state_dim}, "
                f"observed {observed_dim}"
            )
        if int(val_data["physics_states"].shape[1]) != expected_state_dim:
            raise RuntimeError(f"{task_id}: train/validation state dimensions differ")

        train_path = os.path.join(args.output_dir, f"dmc_{task_id}_train.npz")
        val_path = os.path.join(args.output_dir, f"dmc_{task_id}_val.npz")
        save_dataset_npz(train_data, train_path)
        save_dataset_npz(val_data, val_path)

        print(f"Saved {train_path} ({len(train_data['frames'])} frames)")
        print(f"Saved {val_path} ({len(val_data['frames'])} frames)")
        sample_frames[task_id] = train_data["frames"][0]

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    titles = [
        ("cartpole_balance", "Cartpole Balance\n(State dim: 4)"),
        ("finger_spin", "Finger Spin\n(State dim: 6)"),
        ("cheetah_run", "Cheetah Run\n(State dim: 18)"),
    ]
    for ax, (task_id, title) in zip(axes, titles):
        ax.imshow(sample_frames[task_id])
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_02_dmc_tasks.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.2 -> {fig_path}")


if __name__ == "__main__":
    main()
