"""02_views_and_pairs.py - View generation and contrastive pair pipeline demonstration.

Illustrates how spatial translation (random crop shift) and photometric perturbation
(intensity jitter) transform a single sensory observation into positive view pairs (v_1, v_2),
while distinct temporal instances serve as negative distractors (Figure 3.4).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import load_dataset_npz
from worldmodels.train import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstrate view generation pipeline.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--data-file", type=str, default="data/dmc_cartpole_balance_train.npz", help="Dataset path.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Directory for figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not os.path.exists(args.data_file):
        raise FileNotFoundError(
            f"Dataset cache {args.data_file} not found. Run ch03/01_collect_dmc_data.py first."
        )

    data = load_dataset_npz(args.data_file)
    frames = data["frames"]  # (N, 64, 64, 3)

    # Pick an anchor frame and a distinct negative frame
    anchor_idx = 42
    negative_idx = 142

    anchor_frame = torch.from_numpy(frames[anchor_idx]).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    negative_frame = torch.from_numpy(frames[negative_idx]).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)

    view_1, view_2 = pipeline(anchor_frame)

    os.makedirs(args.figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.6))
    panels = [
        (anchor_frame[0], "Anchor Observation\n$x_i$"),
        (view_1[0], "Augmented View 1\n$v_i^{(1)}$ (Positive)"),
        (view_2[0], "Augmented View 2\n$v_i^{(2)}$ (Positive)"),
        (negative_frame[0], "Distinct Instance\n$x_j$ (Negative)"),
    ]

    for ax, (img_tensor, title) in zip(axes, panels):
        img_np = img_tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        ax.imshow(img_np)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")

    plt.tight_layout()
    fig_path = os.path.join(args.figures_dir, "fig03_04_view_pairs.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 3.4 -> {fig_path}")


if __name__ == "__main__":
    main()
