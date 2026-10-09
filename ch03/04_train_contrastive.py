"""04_train_contrastive.py - Contrastive representation learning with NT-Xent.

Trains a ConvEncoder with a two-layer MLP projection head on cartpole_balance
frames using the view augmentation pipeline and NT-Xent loss.

Each seed writes a self-describing metrics file. Figure 3.6 is generated only
when seed outputs match the current experiment configuration exactly, preventing
stale results from older runs from being mixed into a new aggregate.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import evaluate_linear_probe, extract_features
from worldmodels.losses.contrastive import nt_xent_loss
from worldmodels.models.encoders import ConvEncoder, ProjectionHead
from worldmodels.train import get_device, load_json, save_json, set_seed


EXPERIMENT_VERSION = 3
AGGREGATE_SEEDS = (0, 1, 2)
VAL_SAMPLE_CAP = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train contrastive encoder with NT-Xent."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Encoder latent dimension.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="NT-Xent temperature tau.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=5000,
        help="Cap training samples for CPU speed.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data cache directory.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="ch03/figures",
        help="Figures output directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ch03/results",
        help="Results output directory.",
    )
    return parser.parse_args()


def build_experiment_config(args: argparse.Namespace) -> Dict[str, object]:
    """Return the configuration fields that define this experiment run."""
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "task": "cartpole_balance",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "latent_dim": int(args.latent_dim),
        "temperature": float(args.temperature),
        "max_train_samples": int(args.max_train_samples),
        "max_val_samples": VAL_SAMPLE_CAP,
        "sample_selection": "episode_stratified_even_within_episode",
        "augmentation": {
            "max_shift": 3,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
        },
    }


def plot_latent_projection(
    z_untrained: np.ndarray,
    z_trained: np.ndarray,
    physics_states: np.ndarray,
    filepath: str,
) -> None:
    """Plot a 2D PCA projection before and after contrastive training."""
    def _project_2d(z: np.ndarray) -> np.ndarray:
        z_centered = z - np.mean(z, axis=0)
        _, _, vt = np.linalg.svd(
            z_centered,
            full_matrices=False,
        )
        return z_centered @ vt[:2].T

    p_untrained = _project_2d(z_untrained)
    p_trained = _project_2d(z_trained)

    cart_pos = physics_states[:, 0]
    c_norm = (
        (cart_pos - cart_pos.min())
        / (cart_pos.max() - cart_pos.min() + 1e-8)
    )
    gray_shades = [
        str(round(0.15 + 0.7 * value, 2))
        for value in c_norm
    ]

    pole_angle = physics_states[:, 1]
    markers = np.where(
        pole_angle < -0.1,
        "v",
        np.where(pole_angle > 0.1, "^", "o"),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.5, 3.8),
    )

    plot_specs = [
        (
            axes[0],
            p_untrained,
            "Before Training (Random Weights)",
        ),
        (
            axes[1],
            p_trained,
            "After NT-Xent Contrastive Training",
        ),
    ]

    marker_specs = [
        ("v", "Pole Left (< -0.1 rad)"),
        ("o", "Pole Upright ([-0.1, 0.1])"),
        ("^", "Pole Right (> 0.1 rad)"),
    ]

    for ax, projection, title in plot_specs:
        for marker, _label in marker_specs:
            indices = np.where(markers == marker)[0]
            if len(indices) == 0:
                continue

            subset = indices[:150]

            for index in subset:
                ax.scatter(
                    projection[index, 0],
                    projection[index, 1],
                    marker=marker,
                    color=gray_shades[index],
                    edgecolors="black",
                    linewidths=0.5,
                    s=32,
                    alpha=0.85,
                )

        ax.set_title(
            title,
            fontsize=10,
            pad=6,
        )
        ax.set_xlabel(
            "Principal Component 1",
            fontsize=9,
        )
        ax.set_ylabel(
            "Principal Component 2",
            fontsize=9,
        )
        ax.grid(
            True,
            linestyle="--",
            alpha=0.3,
        )

    plt.tight_layout()
    plt.savefig(
        filepath,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def load_compatible_seed_metrics(
    results_dir: str,
    seeds: Tuple[int, ...],
    expected_config: Dict[str, object],
) -> Tuple[
    Dict[int, np.ndarray],
    List[int],
    List[str],
]:
    """Load only seed metrics matching the current experiment configuration.

    Returns:
        compatible: seed -> loss array
        missing: seeds whose files do not exist
        incompatible: human-readable reasons for rejected files
    """
    compatible: Dict[int, np.ndarray] = {}
    missing: List[int] = []
    incompatible: List[str] = []

    expected_epochs = int(expected_config["epochs"])

    for seed in seeds:
        metrics_path = os.path.join(
            results_dir,
            f"contrastive_metrics_seed_{seed}.json",
        )

        if not os.path.exists(metrics_path):
            missing.append(seed)
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception as exc:
            incompatible.append(
                f"seed {seed}: could not read JSON ({exc})"
            )
            continue

        stored_config = metrics.get("experiment")

        if stored_config != expected_config:
            stored_epochs = metrics.get("epochs", "missing")
            stored_experiment_version = (
                stored_config.get("experiment_version")
                if isinstance(stored_config, dict)
                else "missing"
            )
            incompatible.append(
                f"seed {seed}: incompatible configuration "
                f"(stored epochs={stored_epochs}, "
                f"stored experiment_version={stored_experiment_version}, "
                f"expected epochs={expected_epochs}, "
                f"expected experiment_version="
                f"{expected_config['experiment_version']})"
            )
            continue

        stored_seed = metrics.get("seed")
        if stored_seed != seed:
            incompatible.append(
                f"seed {seed}: file declares seed={stored_seed}"
            )
            continue

        losses = metrics.get("losses")
        if not isinstance(losses, list):
            incompatible.append(
                f"seed {seed}: losses is not a list"
            )
            continue

        if len(losses) != expected_epochs:
            incompatible.append(
                f"seed {seed}: losses has {len(losses)} epochs; "
                f"expected {expected_epochs}"
            )
            continue

        losses_array = np.asarray(
            losses,
            dtype=np.float64,
        )

        if losses_array.ndim != 1:
            incompatible.append(
                f"seed {seed}: losses is not one-dimensional"
            )
            continue

        if not np.isfinite(losses_array).all():
            incompatible.append(
                f"seed {seed}: losses contains non-finite values"
            )
            continue

        compatible[seed] = losses_array

    return compatible, missing, incompatible


def maybe_plot_three_seed_loss_band(
    args: argparse.Namespace,
    experiment_config: Dict[str, object],
) -> None:
    """Generate Figure 3.6 only from a complete compatible seed set."""
    compatible, missing, incompatible = load_compatible_seed_metrics(
        args.results_dir,
        AGGREGATE_SEEDS,
        experiment_config,
    )

    if missing or incompatible or len(compatible) != len(AGGREGATE_SEEDS):
        available = sorted(compatible.keys())

        print(
            "Figure 3.6 not generated yet: "
            f"compatible seeds={available}, "
            f"required seeds={list(AGGREGATE_SEEDS)}."
        )

        if missing:
            print(
                "  Missing seed metrics: "
                + ", ".join(str(seed) for seed in missing)
            )

        for reason in incompatible:
            print(f"  Rejected stale/incompatible result: {reason}")

        print(
            "  Run all required seeds with the same experiment "
            "configuration before generating the aggregate plot."
        )
        return

    ordered_losses = [
        compatible[seed]
        for seed in AGGREGATE_SEEDS
    ]

    loss_matrix = np.stack(
        ordered_losses,
        axis=0,
    )

    mean_loss = np.mean(
        loss_matrix,
        axis=0,
    )
    std_loss = np.std(
        loss_matrix,
        axis=0,
    )

    epoch_axis = np.arange(
        1,
        len(mean_loss) + 1,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 3.8),
    )

    ax.plot(
        epoch_axis,
        mean_loss,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="NT-Xent Loss (Mean)",
    )

    ax.fill_between(
        epoch_axis,
        mean_loss - std_loss,
        mean_loss + std_loss,
        color="black",
        alpha=0.15,
        label="±1 Std (3 Seeds)",
    )

    ax.set_title(
        "Contrastive Pre-training Loss on DMC Cartpole Frames",
        fontsize=10,
        pad=8,
    )
    ax.set_xlabel(
        "Epoch",
        fontsize=9,
    )
    ax.set_ylabel(
        "NT-Xent Loss",
        fontsize=9,
    )
    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )
    ax.legend(
        frameon=True,
        fontsize=9,
    )

    plt.tight_layout()

    fig_path = os.path.join(
        args.figures_dir,
        "fig03_06_contrastive_loss.png",
    )

    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary_path = os.path.join(
        args.results_dir,
        "contrastive_loss_summary.json",
    )

    save_json(
        {
            "experiment": experiment_config,
            "seeds": list(AGGREGATE_SEEDS),
            "mean_losses": mean_loss.tolist(),
            "std_losses": std_loss.tolist(),
        },
        summary_path,
    )

    print(
        f"Saved Figure 3.6 (3-seed band) -> {fig_path}"
    )


def main() -> None:
    args = parse_args()

    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.latent_dim <= 0:
        raise ValueError("latent-dim must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.max_train_samples <= 0:
        raise ValueError("max-train-samples must be positive")

    set_seed(args.seed)
    device = get_device()

    experiment_config = build_experiment_config(args)

    train_file = os.path.join(
        args.data_dir,
        "dmc_cartpole_balance_train.npz",
    )
    val_file = os.path.join(
        args.data_dir,
        "dmc_cartpole_balance_val.npz",
    )

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"Training dataset not found at {train_file}. "
            "Run ch03/01_collect_dmc_data.py first."
        )

    if not os.path.exists(val_file):
        raise FileNotFoundError(
            f"Validation dataset not found at {val_file}. "
            "Run ch03/01_collect_dmc_data.py first."
        )

    train_raw = load_dataset_npz(train_file)
    val_raw = load_dataset_npz(val_file)

    n_train = min(
        len(train_raw["frames"]),
        args.max_train_samples,
    )
    n_val = min(
        len(val_raw["frames"]),
        VAL_SAMPLE_CAP,
    )

    if n_train <= 0:
        raise ValueError("Training dataset contains no usable frames")

    if n_val <= 0:
        raise ValueError("Validation dataset contains no usable frames")

    train_indices = select_episode_stratified_indices(
        train_raw["episode_ids"],
        n_train,
    )
    val_indices = select_episode_stratified_indices(
        val_raw["episode_ids"],
        n_val,
    )

    train_dataset = DMCDataset(
        frames=train_raw["frames"],
        physics_states=train_raw["physics_states"],
        episode_ids=train_raw["episode_ids"],
        sample_indices=train_indices,
    )

    val_dataset = DMCDataset(
        frames=val_raw["frames"],
        physics_states=val_raw["physics_states"],
        episode_ids=val_raw["episode_ids"],
        sample_indices=val_indices,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    encoder = ConvEncoder(
        in_channels=3,
        latent_dim=args.latent_dim,
    ).to(device)

    proj_head = ProjectionHead(
        in_dim=args.latent_dim,
        hidden_dim=64,
        out_dim=args.latent_dim,
    ).to(device)

    z_init, y_val_phys = extract_features(
        encoder,
        val_loader,
        device,
    )

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(proj_head.parameters()),
        lr=args.lr,
    )

    view_pipeline = ViewPipeline(
        max_shift=3,
        brightness_range=0.1,
        contrast_range=0.1,
    )

    epoch_losses: List[float] = []

    print(
        f"Starting contrastive training "
        f"(seed={args.seed}, "
        f"epochs={args.epochs}, "
        f"device={device})..."
    )

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        proj_head.train()

        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            images = batch["image"].to(device)

            view_1, view_2 = view_pipeline(images)

            z1 = encoder(view_1)
            z2 = encoder(view_2)

            p1 = proj_head(z1)
            p2 = proj_head(z2)

            loss = nt_xent_loss(
                p1,
                p2,
                temperature=args.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(images)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

        if total_examples == 0:
            raise RuntimeError(
                "Training DataLoader produced zero examples"
            )

        average_loss = total_loss / total_examples
        epoch_losses.append(float(average_loss))

        if epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"  Epoch {epoch:02d}/{args.epochs:02d} | "
                f"NT-Xent Loss: {average_loss:.4f}"
            )

    probe_results = evaluate_linear_probe(
        encoder,
        train_loader,
        val_loader,
        device,
    )

    print(
        f"Linear Probe Mean R^2: "
        f"{probe_results['mean_r2']:.4f}"
    )

    print(
        "R^2 per variable "
        "(cart_pos, pole_angle, cart_vel, ang_vel): "
        f"{probe_results['r2_per_variable']}"
    )

    z_trained, _ = extract_features(
        encoder,
        val_loader,
        device,
    )

    os.makedirs(
        args.checkpoints_dir,
        exist_ok=True,
    )
    os.makedirs(
        args.figures_dir,
        exist_ok=True,
    )
    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        args.checkpoints_dir,
        f"contrastive_encoder_seed_{args.seed}.pt",
    )

    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "latent_dim": args.latent_dim,
            "seed": args.seed,
            "experiment": experiment_config,
        },
        checkpoint_path,
    )

    metrics_path = os.path.join(
        args.results_dir,
        f"contrastive_metrics_seed_{args.seed}.json",
    )

    save_json(
        {
            "experiment": experiment_config,
            "seed": args.seed,
            "epochs": args.epochs,
            "losses": epoch_losses,
            "final_loss": epoch_losses[-1],
            "probe_mean_r2": probe_results["mean_r2"],
            "probe_r2_per_variable": probe_results[
                "r2_per_variable"
            ],
        },
        metrics_path,
    )

    fig07_path = os.path.join(
        args.figures_dir,
        f"fig03_07_latent_projection_seed_{args.seed}.png",
    )

    plot_latent_projection(
        z_init,
        z_trained,
        y_val_phys,
        fig07_path,
    )

    print(
        f"Saved Figure 3.7 -> {fig07_path}"
    )

    maybe_plot_three_seed_loss_band(
        args,
        experiment_config,
    )


if __name__ == "__main__":
    main()