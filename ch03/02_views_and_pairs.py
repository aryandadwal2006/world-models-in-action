"""02_views_and_pairs.py - Visualize the Chapter 3 contrastive pair pipeline."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import torch

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import load_dataset_npz
from worldmodels.train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate contrastive view generation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/dmc_cartpole_balance_train.npz",
    )
    parser.add_argument("--anchor-index", type=int, default=42)
    parser.add_argument("--figures-dir", type=str, default="ch03/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = load_dataset_npz(args.data_file)
    frames = data["frames"]
    episode_ids = data["episode_ids"]
    anchor_idx = args.anchor_index
    if not (0 <= anchor_idx < len(frames)):
        raise IndexError("anchor-index is outside the dataset")

    anchor_episode = episode_ids[anchor_idx]
    different_episode = [
        int(i) for i in range(len(frames)) if episode_ids[i] != anchor_episode
    ]
    if not different_episode:
        raise RuntimeError("Dataset contains no frame from a different episode")
    negative_idx = different_episode[0]

    anchor = (
        torch.from_numpy(frames[anchor_idx])
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )
    negative = (
        torch.from_numpy(frames[negative_idx])
        .permute(2, 0, 1)
        .float()
        .unsqueeze(0)
        / 255.0
    )

    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)
    view_1, view_2 = pipeline(anchor)

    os.makedirs(args.figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    panels = [
        (anchor[0], "Anchor Observation\n$x_i$"),
        (view_1[0], "Augmented View 1\n$v_i^{(1)}$"),
        (view_2[0], "Augmented View 2\n$v_i^{(2)}$"),
        (negative[0], "Different Episode\n$x_j$"),
    ]
    for ax, (image, title) in zip(axes, panels):
        ax.imshow(image.permute(1, 2, 0).clamp(0, 1).numpy())
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_04_view_pairs.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.4 -> {fig_path}")


if __name__ == "__main__":
    main()
