"""Compare representation objectives for Table 3.4."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    ),
)

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
from worldmodels.eval.probes import (
    expanded_state_dim,
    evaluate_linear_probe,
    infer_angular_position_indices,
    transform_state_targets_np,
    transform_state_targets_torch,
)
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

EXPERIMENT_VERSION = 3

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
    )

    return parser.parse_args()


def make_signature(
    args,
    dataset_metadata,
):
    """Identify every choice that affects Table 3.4."""
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
        "seeds": [
            int(seed)
            for seed in args.seeds
        ],
        "tasks": [
            name
            for name, _ in REQUIRED_TASKS
        ],
        "augmentation": {
            "max_shift": 3,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
        },
        "probe": {
            "feature_standardization": True,
            "angular_position_encoding": "sin_cos",
            "angular_r2": "circular_chordal",
        },
        "datasets": dataset_metadata,
    }


def empty_progress(
    signature,
):
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


def load_progress(
    path,
    signature,
    fresh,
):
    if (
        fresh
        or not os.path.exists(path)
    ):
        return empty_progress(
            signature
        )

    try:
        progress = load_json(
            path
        )
    except Exception:
        print(
            "Existing Table 3.4 progress "
            "could not be read; "
            "starting a fresh run."
        )

        return empty_progress(
            signature
        )

    if (
        progress.get("experiment")
        != signature
    ):
        print(
            "Existing Table 3.4 progress "
            "is incompatible; "
            "starting a fresh run."
        )

        return empty_progress(
            signature
        )

    print(
        f"Resuming compatible "
        f"Table 3.4 progress -> {path}"
    )

    return progress


def save_progress(
    progress,
    path,
):
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

    return (
        train_loader,
        val_loader,
    )


def summarize_probe(
    result,
    state_dim,
):
    r2 = np.asarray(
        result["r2_per_variable"],
        dtype=np.float64,
    )

    if r2.shape != (state_dim,):
        raise ValueError(
            f"Expected {state_dim} state R2 values, "
            f"got {r2.shape}"
        )

    n_position = state_dim // 2

    return {
        "r2_per_variable": r2.tolist(),
        "mean_r2": float(
            np.mean(r2)
        ),
        "position_mean_r2": float(
            np.mean(
                r2[:n_position]
            )
        ),
        "velocity_mean_r2": float(
            np.mean(
                r2[n_position:]
            )
        ),
    }


def transformed_training_statistics(
    train_loader,
    angular_indices,
):
    """Compute train statistics in the transformed supervised target space."""
    states = np.asarray(
        train_loader.dataset.physics_states,
        dtype=np.float64,
    )

    targets = transform_state_targets_np(
        states,
        angular_indices,
    )

    mean = targets.mean(
        axis=0
    )

    std = np.maximum(
        targets.std(
            axis=0
        ),
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
    """Train the privileged supervised state-prediction reference."""
    set_seed(seed)

    angular_indices = (
        infer_angular_position_indices(
            state_dim
        )
    )

    target_dim = expanded_state_dim(
        state_dim,
        angular_indices,
    )

    encoder = ConvEncoder(
        latent_dim=args.latent_dim
    ).to(device)

    head = nn.Linear(
        args.latent_dim,
        target_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(
            encoder.parameters()
        )
        + list(
            head.parameters()
        ),
        lr=args.lr,
    )

    target_mean, target_std = (
        transformed_training_statistics(
            train_loader,
            angular_indices,
        )
    )

    target_mean = target_mean.to(
        device
    )

    target_std = target_std.to(
        device
    )

    for _ in range(
        args.epochs
    ):
        encoder.train()
        head.train()

        for batch in train_loader:
            x = batch[
                "image"
            ].to(device)

            y = batch[
                "physics_state"
            ].to(device)

            target = (
                transform_state_targets_torch(
                    y,
                    angular_indices,
                )
            )

            target = (
                target - target_mean
            ) / target_std

            pred = head(
                encoder(x)
            )

            loss = nn.functional.mse_loss(
                pred,
                target,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return summarize_probe(
        evaluate_linear_probe(
            encoder,
            train_loader,
            val_loader,
            device,
        ),
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
    """Train the NT-Xent reference."""
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
        list(
            encoder.parameters()
        )
        + list(
            projector.parameters()
        ),
        lr=args.lr,
    )

    pipeline = ViewPipeline(
        max_shift=3,
        brightness_range=0.1,
        contrast_range=0.1,
    )

    for _ in range(
        args.epochs
    ):
        encoder.train()
        projector.train()

        for batch in train_loader:
            x = batch[
                "image"
            ].to(device)

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

    return summarize_probe(
        evaluate_linear_probe(
            encoder,
            train_loader,
            val_loader,
            device,
        ),
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
    """Train the VICReg reference."""
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
        list(
            encoder.parameters()
        )
        + list(
            projector.parameters()
        ),
        lr=args.lr,
    )

    pipeline = ViewPipeline(
        max_shift=3,
        brightness_range=0.1,
        contrast_range=0.1,
    )

    for _ in range(
        args.epochs
    ):
        encoder.train()
        projector.train()

        for batch in train_loader:
            x = batch[
                "image"
            ].to(device)

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

    return summarize_probe(
        evaluate_linear_probe(
            encoder,
            train_loader,
            val_loader,
            device,
        ),
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
    """Train the MAE reference."""
    set_seed(seed)

    model = SimpleMAE(
        latent_dim=args.latent_dim,
        mask_ratio=0.75,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    for _ in range(
        args.epochs
    ):
        model.train()

        for batch in train_loader:
            x = batch[
                "image"
            ].to(device)

            loss, _, _ = model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return summarize_probe(
        evaluate_linear_probe(
            MAEEncoder(model),
            train_loader,
            val_loader,
            device,
        ),
        state_dim,
    )


def build_final_table(
    progress,
    args,
):
    table = {}

    results = progress[
        "results"
    ]

    for task_name, state_dim in (
        REQUIRED_TASKS
    ):
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
                        f"Missing {task_name} / "
                        f"{method} / seed={seed}"
                    )

                seed_results.append(
                    result
                )

            table[
                task_name
            ][method] = {
                "position_mean": float(
                    np.mean(
                        [
                            r[
                                "position_mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "position_std": float(
                    np.std(
                        [
                            r[
                                "position_mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "velocity_mean": float(
                    np.mean(
                        [
                            r[
                                "velocity_mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "velocity_std": float(
                    np.std(
                        [
                            r[
                                "velocity_mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "mean_r2": float(
                    np.mean(
                        [
                            r[
                                "mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "std_r2": float(
                    np.std(
                        [
                            r[
                                "mean_r2"
                            ]
                            for r in seed_results
                        ]
                    )
                ),
                "seed_results": seed_results,
            }

    return table


def complete(
    progress,
    args,
):
    results = progress[
        "results"
    ]

    return all(
        str(seed)
        in results[
            task
        ][method]
        for task, _ in REQUIRED_TASKS
        for method in METHOD_NAMES
        for seed in args.seeds
    )


def main():
    args = parse_args()

    if not args.seeds:
        raise ValueError(
            "At least one seed is required"
        )

    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.lr <= 0
        or args.latent_dim <= 0
    ):
        raise ValueError(
            "Invalid optimization settings"
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

    for task_name, _state_dim in (
        REQUIRED_TASKS
    ):
        train_path = os.path.join(
            args.data_dir,
            f"dmc_{task_name}_train.npz",
        )

        val_path = os.path.join(
            args.data_dir,
            f"dmc_{task_name}_val.npz",
        )

        if (
            not os.path.exists(train_path)
            or not os.path.exists(val_path)
        ):
            raise FileNotFoundError(
                f"Missing dataset for {task_name}"
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

    signature = make_signature(
        args,
        dataset_metadata,
    )

    progress = load_progress(
        progress_path,
        signature,
        args.fresh,
    )

    save_progress(
        progress,
        progress_path,
    )

    task_methods = {
        METHOD_NAMES[0]: train_supervised,
        METHOD_NAMES[1]: train_contrastive,
        METHOD_NAMES[2]: train_vicreg,
        METHOD_NAMES[3]: train_mae,
    }

    for task_name, state_dim in (
        REQUIRED_TASKS
    ):
        train_raw, val_raw = (
            dataset_cache[
                task_name
            ]
        )

        n_train = min(
            len(train_raw["frames"]),
            args.max_train_samples,
        )

        n_val = min(
            len(val_raw["frames"]),
            args.max_val_samples,
        )

        train_ds = DMCDataset(
            train_raw["frames"][
                :n_train
            ],
            train_raw["physics_states"][
                :n_train
            ],
            train_raw["episode_ids"][
                :n_train
            ],
        )

        val_ds = DMCDataset(
            val_raw["frames"][
                :n_val
            ],
            val_raw["physics_states"][
                :n_val
            ],
            val_raw["episode_ids"][
                :n_val
            ],
        )

        for method in METHOD_NAMES:
            for seed in args.seeds:
                key = str(seed)

                existing = (
                    progress[
                        "results"
                    ][task_name][method].get(
                        key
                    )
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

                if method == METHOD_NAMES[0]:
                    result = task_methods[
                        method
                    ](
                        train_loader,
                        val_loader,
                        state_dim,
                        args,
                        device,
                        seed,
                    )
                else:
                    result = task_methods[
                        method
                    ](
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
                    key
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

    if not complete(
        progress,
        args,
    ):
        print(
            "Table 3.4 incomplete; "
            "progress is safely checkpointed."
        )
        return

    table = build_final_table(
        progress,
        args,
    )

    print(
        "\nTable 3.4: "
        "position and velocity probe R^2"
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
        "\nSaved final Table 3.4 results -> "
        "ch03/results/"
        "table_03_04_objective_comparison.json"
    )


if __name__ == "__main__":
    main()