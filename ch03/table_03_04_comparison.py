"""table_03_04_comparison.py - Compare representation objectives.

Compares supervised state prediction, NT-Xent, VICReg, and MAE on cartpole
balance and finger spin.

The experiment checkpoints after every seed so an interrupted long CPU run can
resume without discarding completed measurements.

The supervised reference standardizes its privileged training targets during
optimization. This conditioning is internal to the supervised reference;
self-supervised methods never receive physics-state targets during training.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from worldmodels.data.augment import ViewPipeline
from worldmodels.data.dmc_data import (
    DMCDataset,
    load_dataset_npz,
    metadata_from_dataset,
)
from worldmodels.eval.probes import evaluate_linear_probe
from worldmodels.losses.contrastive import (
    nt_xent_loss,
    vicreg_loss,
)
from worldmodels.models.encoders import (
    ConvEncoder,
    MAEEncoder,
    ProjectionHead,
    SimpleMAE,
)
from worldmodels.train import (
    get_device,
    load_json,
    save_json,
    set_seed,
)


EXPERIMENT_VERSION = 2
POSITION_VARIABLES = 2
REQUIRED_TASKS = (
    ("cartpole_balance", 4),
    ("finger_spin", 6),
)
METHOD_NAMES = (
    "Supervised State-Prediction Reference",
    "Contrastive (NT-Xent)",
    "VICReg",
    "Masked Autoencoder (MAE)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Table 3.4."
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=3500,
    )

    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="ch03/results",
    )

    parser.add_argument(
        "--progress-file",
        type=str,
        default="table_03_04_progress.json",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard any compatible progress file and start over.",
    )

    return parser.parse_args()


def make_experiment_signature(
    args: argparse.Namespace,
    dataset_metadata: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    """Return everything needed to identify a Table 3.4 run."""
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "latent_dim": int(args.latent_dim),
        "max_train_samples": int(
            args.max_train_samples
        ),
        "max_val_samples": int(
            args.max_val_samples
        ),
        "seeds": [int(seed) for seed in args.seeds],
        "tasks": [task for task, _ in REQUIRED_TASKS],
        "augmentation": {
            "max_shift": 3,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
        },
        "datasets": dataset_metadata,
    }


def empty_progress(signature: Dict[str, object]) -> Dict[str, object]:
    return {
        "experiment": signature,
        "results": {
            task: {
                method: {}
                for method in METHOD_NAMES
            }
            for task, _ in REQUIRED_TASKS
        },
    }


def load_or_initialize_progress(
    path: str,
    signature: Dict[str, object],
    fresh: bool,
) -> Dict[str, object]:
    """Load compatible progress or create a new checkpoint."""
    if fresh or not os.path.exists(path):
        return empty_progress(signature)

    try:
        progress = load_json(path)
    except Exception:
        print(
            "Existing Table 3.4 progress could not be read; "
            "starting a fresh run."
        )
        return empty_progress(signature)

    if progress.get("experiment") != signature:
        print(
            "Existing Table 3.4 progress is from a different "
            "experiment configuration; starting a fresh run."
        )
        return empty_progress(signature)

    print(
        f"Resuming compatible Table 3.4 progress -> {path}"
    )

    return progress


def save_progress(
    progress: Dict[str, object],
    path: str,
) -> None:
    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    save_json(
        progress,
        path,
    )


def make_loaders(
    train_ds,
    val_ds,
    batch_size,
    seed,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


def summarize_probe(
    result: dict,
    state_dim: int,
) -> dict:
    r2 = np.asarray(
        result["r2_per_variable"],
        dtype=np.float64,
    )

    if r2.shape != (state_dim,):
        raise ValueError(
            f"Expected {state_dim} R^2 values, got {r2.shape}"
        )

    n_position = state_dim // 2

    position = float(
        np.mean(
            r2[:n_position]
        )
    )

    velocity = float(
        np.mean(
            r2[n_position:]
        )
    )

    return {
        "r2_per_variable": r2.tolist(),
        "mean_r2": float(
            np.mean(r2)
        ),
        "position_mean_r2": position,
        "velocity_mean_r2": velocity,
    }


def training_target_statistics(
    train_loader,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute target standardization statistics from privileged train labels."""
    dataset = train_loader.dataset

    states = np.asarray(
        dataset.physics_states,
        dtype=np.float64,
    )

    mean = states.mean(
        axis=0
    )

    std = states.std(
        axis=0
    )

    std = np.maximum(
        std,
        1e-6,
    )

    return (
        torch.from_numpy(
            mean
        ).float(),
        torch.from_numpy(
            std
        ).float(),
    )


def train_supervised(
    train_loader,
    val_loader,
    state_dim,
    args,
    device,
    seed,
):
    """Train the supervised reference with standardized targets."""
    set_seed(seed)

    encoder = ConvEncoder(
        latent_dim=args.latent_dim
    ).to(device)

    head = nn.Linear(
        args.latent_dim,
        state_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(head.parameters()),
        lr=args.lr,
    )

    target_mean, target_std = (
        training_target_statistics(
            train_loader
        )
    )

    target_mean = target_mean.to(device)
    target_std = target_std.to(device)

    for _ in range(args.epochs):
        encoder.train()
        head.train()

        for batch in train_loader:
            x = batch["image"].to(device)

            y = batch[
                "physics_state"
            ].to(device)

            y_scaled = (
                y - target_mean
            ) / target_std

            pred = head(
                encoder(x)
            )

            loss = nn.functional.mse_loss(
                pred,
                y_scaled,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    result = evaluate_linear_probe(
        encoder,
        train_loader,
        val_loader,
        device,
    )

    return summarize_probe(
        result,
        state_dim,
    )


def train_contrastive(
    train_loader,
    val_loader,
    args,
    device,
    seed,
    state_dim,
):
    """Train the NT-Xent representation reference."""
    set_seed(seed)

    encoder = ConvEncoder(
        latent_dim=args.latent_dim
    ).to(device)

    projector = ProjectionHead(
        args.latent_dim,
        64,
        args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(projector.parameters()),
        lr=args.lr,
    )

    pipeline = ViewPipeline(
        max_shift=3,
        brightness_range=0.1,
        contrast_range=0.1,
    )

    for _ in range(args.epochs):
        encoder.train()
        projector.train()

        for batch in train_loader:
            x = batch["image"].to(device)

            v1, v2 = pipeline(x)

            loss = nt_xent_loss(
                projector(
                    encoder(v1)
                ),
                projector(
                    encoder(v2)
                ),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    result = evaluate_linear_probe(
        encoder,
        train_loader,
        val_loader,
        device,
    )

    return summarize_probe(
        result,
        state_dim,
    )


def train_vicreg(
    train_loader,
    val_loader,
    args,
    device,
    seed,
    state_dim,
):
    """Train the VICReg representation reference."""
    set_seed(seed)

    encoder = ConvEncoder(
        latent_dim=args.latent_dim
    ).to(device)

    projector = ProjectionHead(
        args.latent_dim,
        64,
        args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters())
        + list(projector.parameters()),
        lr=args.lr,
    )

    pipeline = ViewPipeline(
        max_shift=3,
        brightness_range=0.1,
        contrast_range=0.1,
    )

    for _ in range(args.epochs):
        encoder.train()
        projector.train()

        for batch in train_loader:
            x = batch["image"].to(device)

            v1, v2 = pipeline(x)

            loss, _, _, _ = vicreg_loss(
                projector(
                    encoder(v1)
                ),
                projector(
                    encoder(v2)
                ),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    result = evaluate_linear_probe(
        encoder,
        train_loader,
        val_loader,
        device,
    )

    return summarize_probe(
        result,
        state_dim,
    )


def train_mae(
    train_loader,
    val_loader,
    args,
    device,
    seed,
    state_dim,
):
    """Train and evaluate the masked autoencoder reference."""
    set_seed(seed)

    model = SimpleMAE(
        latent_dim=args.latent_dim,
        mask_ratio=0.75,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    for _ in range(args.epochs):
        model.train()

        for batch in train_loader:
            x = batch["image"].to(device)

            loss, _, _ = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    result = evaluate_linear_probe(
        MAEEncoder(model),
        train_loader,
        val_loader,
        device,
    )

    return summarize_probe(
        result,
        state_dim,
    )


def build_final_table(
    progress: Dict[str, object],
    args,
) -> Dict[str, object]:
    """Convert per-seed progress into the manuscript-facing table schema."""
    table = {}

    results = progress["results"]

    for task_name, state_dim in REQUIRED_TASKS:
        table[task_name] = {}

        for method in METHOD_NAMES:
            seed_results = []

            for seed in args.seeds:
                result = results[
                    task_name
                ][method].get(
                    str(seed)
                )

                if result is None:
                    raise RuntimeError(
                        f"Missing result for "
                        f"{task_name} / "
                        f"{method} / "
                        f"seed={seed}"
                    )

                seed_results.append(result)

            table[task_name][method] = {
                "position_mean": float(
                    np.mean(
                        [
                            r["position_mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "position_std": float(
                    np.std(
                        [
                            r["position_mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "velocity_mean": float(
                    np.mean(
                        [
                            r["velocity_mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "velocity_std": float(
                    np.std(
                        [
                            r["velocity_mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "mean_r2": float(
                    np.mean(
                        [
                            r["mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "std_r2": float(
                    np.std(
                        [
                            r["mean_r2"]
                            for r in seed_results
                        ]
                    )
                ),
                "seed_results": seed_results,
            }

    return table


def print_current_table(
    progress: Dict[str, object],
    args,
) -> None:
    """Print only completed rows currently available."""
    results = progress["results"]

    print("\nCompleted Table 3.4 measurements")

    for task_name, _state_dim in REQUIRED_TASKS:
        for method in METHOD_NAMES:
            completed = results[
                task_name
            ][method]

            if not completed:
                continue

            for seed in args.seeds:
                result = completed.get(
                    str(seed)
                )

                if result is None:
                    continue

                print(
                    f"{task_name} | "
                    f"{method} | "
                    f"seed={seed} | "
                    f"position="
                    f"{result['position_mean_r2']:.3f} | "
                    f"velocity="
                    f"{result['velocity_mean_r2']:.3f}"
                )


def all_results_complete(
    progress: Dict[str, object],
    args,
) -> bool:
    results = progress["results"]

    for task_name, _state_dim in REQUIRED_TASKS:
        for method in METHOD_NAMES:
            for seed in args.seeds:
                if str(seed) not in results[
                    task_name
                ][method]:
                    return False

    return True


def main() -> None:
    args = parse_args()

    if not args.seeds:
        raise ValueError(
            "At least one seed is required"
        )

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive"
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be positive"
        )

    if args.latent_dim <= 0:
        raise ValueError(
            "latent-dim must be positive"
        )

    if args.max_train_samples <= 0:
        raise ValueError(
            "max-train-samples must be positive"
        )

    if args.max_val_samples <= 0:
        raise ValueError(
            "max-val-samples must be positive"
        )

    device = get_device()

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    progress_path = os.path.join(
        args.results_dir,
        args.progress_file,
    )

    dataset_cache = {}

    dataset_metadata = {}

    for task_name, _state_dim in REQUIRED_TASKS:
        train_path = os.path.join(
            args.data_dir,
            f"dmc_{task_name}_train.npz",
        )

        val_path = os.path.join(
            args.data_dir,
            f"dmc_{task_name}_val.npz",
        )

        if not os.path.exists(train_path):
            raise FileNotFoundError(
                f"Missing dataset: {train_path}"
            )

        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Missing dataset: {val_path}"
            )

        train_raw = load_dataset_npz(
            train_path
        )

        val_raw = load_dataset_npz(
            val_path
        )

        dataset_metadata[
            task_name
        ] = {
            "train": metadata_from_dataset(
                train_raw
            ),
            "validation": metadata_from_dataset(
                val_raw
            ),
        }

        dataset_cache[
            task_name
        ] = (
            train_raw,
            val_raw,
        )

    signature = make_experiment_signature(
        args,
        dataset_metadata,
    )

    progress = load_or_initialize_progress(
        progress_path,
        signature,
        args.fresh,
    )

    if args.fresh:
        save_progress(
            progress,
            progress_path,
        )

    task_methods = {
        "Supervised State-Prediction Reference": train_supervised,
        "Contrastive (NT-Xent)": train_contrastive,
        "VICReg": train_vicreg,
        "Masked Autoencoder (MAE)": train_mae,
    }

    for task_name, state_dim in REQUIRED_TASKS:
        train_raw, val_raw = dataset_cache[
            task_name
        ]

        n_train = min(
            len(train_raw["frames"]),
            args.max_train_samples,
        )

        n_val = min(
            len(val_raw["frames"]),
            args.max_val_samples,
        )

        train_ds = DMCDataset(
            train_raw["frames"][:n_train],
            train_raw["physics_states"][:n_train],
            train_raw["episode_ids"][:n_train],
        )

        val_ds = DMCDataset(
            val_raw["frames"][:n_val],
            val_raw["physics_states"][:n_val],
            val_raw["episode_ids"][:n_val],
        )

        for method in METHOD_NAMES:
            train_fn = task_methods[
                method
            ]

            for seed in args.seeds:
                seed_key = str(seed)

                existing = progress[
                    "results"
                ][task_name][method].get(
                    seed_key
                )

                if existing is not None:
                    print(
                        f"RESUME | "
                        f"{task_name} | "
                        f"{method} | "
                        f"seed={seed} | "
                        f"position="
                        f"{existing['position_mean_r2']:.3f} | "
                        f"velocity="
                        f"{existing['velocity_mean_r2']:.3f}"
                    )
                    continue

                print(
                    f"\nTraining | "
                    f"{task_name} | "
                    f"{method} | "
                    f"seed={seed}"
                )

                train_loader, val_loader = (
                    make_loaders(
                        train_ds,
                        val_ds,
                        args.batch_size,
                        seed,
                    )
                )

                if method == (
                    "Supervised State-Prediction Reference"
                ):
                    result = train_fn(
                        train_loader,
                        val_loader,
                        state_dim,
                        args,
                        device,
                        seed,
                    )
                else:
                    result = train_fn(
                        train_loader,
                        val_loader,
                        args,
                        device,
                        seed,
                        state_dim,
                    )

                progress[
                    "results"
                ][task_name][method][
                    seed_key
                ] = result

                save_progress(
                    progress,
                    progress_path,
                )

                print(
                    f"COMPLETED | "
                    f"{task_name} | "
                    f"{method} | "
                    f"seed={seed} | "
                    f"position="
                    f"{result['position_mean_r2']:.3f} | "
                    f"velocity="
                    f"{result['velocity_mean_r2']:.3f}"
                )

    print_current_table(
        progress,
        args,
    )

    if not all_results_complete(
        progress,
        args,
    ):
        completed = 0
        total = (
            len(REQUIRED_TASKS)
            * len(METHOD_NAMES)
            * len(args.seeds)
        )

        for task_name, _state_dim in REQUIRED_TASKS:
            for method in METHOD_NAMES:
                completed += len(
                    progress[
                        "results"
                    ][task_name][method]
                )

        print(
            f"\nTable 3.4 incomplete: "
            f"{completed}/{total} seed-runs complete."
        )

        return

    table = build_final_table(
        progress,
        args,
    )

    print(
        "\nTable 3.4: position and velocity probe R^2"
    )

    for method in METHOD_NAMES:
        cp = table[
            "cartpole_balance"
        ][method]

        fg = table[
            "finger_spin"
        ][method]

        print(
            f"{method:<38} | "
            f"CP pos "
            f"{cp['position_mean']:.3f} ± "
            f"{cp['position_std']:.3f} | "
            f"CP vel "
            f"{cp['velocity_mean']:.3f} ± "
            f"{cp['velocity_std']:.3f} | "
            f"FG pos "
            f"{fg['position_mean']:.3f} ± "
            f"{fg['position_std']:.3f} | "
            f"FG vel "
            f"{fg['velocity_mean']:.3f} ± "
            f"{fg['velocity_std']:.3f}"
        )

    final_path = os.path.join(
        args.results_dir,
        "table_03_04_objective_comparison.json",
    )

    save_json(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "config": vars(args),
            "experiment": signature,
            "data": table,
        },
        final_path,
    )

    print(
        f"\nSaved final Table 3.4 results -> {final_path}"
    )


if __name__ == "__main__":
    main()