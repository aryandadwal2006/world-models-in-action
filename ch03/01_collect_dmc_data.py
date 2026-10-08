"""01_collect_dmc_data.py - Data collection and offline caching for DMC benchmark tasks.

Interacts with DeepMind Control Suite (cartpole_balance, finger_spin, cheetah_run)
under a uniform random policy, renders 64x64 pixel frames, records privileged physics states
strictly for downstream evaluation, and saves cached .npz files alongside Figure 3.2.
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure repository root is in Python module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from worldmodels.data.dmc_data import collect_task_dataset, save_dataset_npz
from worldmodels.train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect and cache DMC datasets.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment rollouts.")
    parser.add_argument("--num-train-frames", type=int, default=10000, help="Train frames per task.")
    parser.add_argument("--num-val-frames", type=int, default=2000, help="Val frames per task.")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save .npz caches.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Directory for figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    tasks = [
        ("cartpole", "balance"),
        ("finger", "spin"),
        ("cheetah", "run"),
    ]

    sample_frames = {}

    for domain, task in tasks:
        task_id = f"{domain}_{task}"
        print(f"Collecting DMC rollouts for {task_id} (seed={args.seed})...")

        total_frames = args.num_train_frames + args.num_val_frames
        full_data = collect_task_dataset(
            domain_name=domain,
            task_name=task,
            num_frames=total_frames,
            seed=args.seed,
            image_size=(64, 64),
        )

        # Split into training and validation sets along frame sequence
        train_slice = slice(0, args.num_train_frames)
        val_slice = slice(args.num_train_frames, total_frames)

        train_dict = {
            "frames": full_data["frames"][train_slice],
            "physics_states": full_data["physics_states"][train_slice],
            "actions": full_data["actions"][train_slice],
            "rewards": full_data["rewards"][train_slice],
            "dones": full_data["dones"][train_slice],
            "episode_ids": full_data["episode_ids"][train_slice],
            "domain_name": full_data["domain_name"],
            "task_name": full_data["task_name"],
        }

        val_dict = {
            "frames": full_data["frames"][val_slice],
            "physics_states": full_data["physics_states"][val_slice],
            "actions": full_data["actions"][val_slice],
            "rewards": full_data["rewards"][val_slice],
            "dones": full_data["dones"][val_slice],
            "episode_ids": full_data["episode_ids"][val_slice],
            "domain_name": full_data["domain_name"],
            "task_name": full_data["task_name"],
        }

        train_path = os.path.join(args.output_dir, f"dmc_{task_id}_train.npz")
        val_path = os.path.join(args.output_dir, f"dmc_{task_id}_val.npz")

        save_dataset_npz(train_dict, train_path)
        save_dataset_npz(val_dict, val_path)
        print(f"Saved {task_id} train -> {train_path} ({len(train_dict['frames'])} frames)")
        print(f"Saved {task_id} val   -> {val_path} ({len(val_dict['frames'])} frames)")

        # Keep first frame for figure generation
        sample_frames[task_id] = full_data["frames"][0]

    # Generate Figure 3.2: 64x64 rendered frames for each of the three tasks
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    task_titles = [
        ("cartpole_balance", "Cartpole Balance\n(State dim: 4)"),
        ("finger_spin", "Finger Spin\n(State dim: 6)"),
        ("cheetah_run", "Cheetah Run\n(State dim: 18)"),
    ]

    for ax, (task_id, title) in zip(axes, task_titles):
        img = sample_frames[task_id]
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_02_dmc_tasks.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.2 -> {fig_path}")


if __name__ == "__main__":
    main()
