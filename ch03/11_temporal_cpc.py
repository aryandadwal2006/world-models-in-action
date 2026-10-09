"""Temporal CPC and missing-observation tests for Chapter 3.

The CPC objective uses a causal GRU context and bilinear InfoNCE compatibility.
Latent representations and predicted latents are L2-normalized. State probes
use standardized features and circular evaluation for angular positions.

The evaluation distinguishes current-state decoding, next-state prediction,
and prediction through a real unobserved interval. Angular metrics respect the
period of each rendered state variable.
"""

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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from worldmodels.data.dmc_data import (
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import (
    StateLinearProbe,
    compute_structured_r2_score,
    infer_angular_position_indices,
)
from worldmodels.models.encoders import ConvEncoder
from worldmodels.train import (
    get_device,
    load_json,
    save_json,
    set_seed,
)

EXPERIMENT_VERSION = 4


class SequenceDataset(Dataset):
    """Yield episode-safe contiguous sequence windows."""

    def __init__(
        self,
        frames,
        physics_states,
        episode_ids,
        seq_len: int,
        max_sequences: int | None = None,
    ):
        if seq_len < 2:
            raise ValueError(
                "seq_len must be >= 2"
            )

        if not (
            len(frames)
            == len(physics_states)
            == len(episode_ids)
        ):
            raise ValueError(
                "frames, states, and episode_ids "
                "must have equal lengths"
            )

        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids
        self.seq_len = seq_len

        self.valid_indices = []

        for start in range(
            len(frames) - seq_len + 1
        ):
            window = episode_ids[
                start : start + seq_len
            ]

            if np.all(
                window == window[0]
            ):
                self.valid_indices.append(start)

        if max_sequences is not None and len(self.valid_indices) > max_sequences:
            self.valid_indices = select_episode_stratified_indices(
                self.episode_ids,
                max_sequences,
                candidate_indices=np.asarray(self.valid_indices, dtype=np.int64),
            ).tolist()

    def __len__(self):
        return len(
            self.valid_indices
        )

    def __getitem__(self, index):
        start = self.valid_indices[index]
        end = start + self.seq_len

        images = (
            torch.from_numpy(
                self.frames[start:end]
            )
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )

        states = torch.from_numpy(
            self.physics_states[start:end]
        ).float()

        return {
            "images": images,
            "physics_states": states,
        }


class TemporalCPC(nn.Module):
    """Causal GRU CPC model with normalized latent prediction."""

    def __init__(
        self,
        encoder,
        latent_dim=16,
        context_dim=32,
        k_steps=3,
        temperature=0.5,
    ):
        super().__init__()

        if min(
            latent_dim,
            context_dim,
            k_steps,
        ) <= 0:
            raise ValueError(
                "latent_dim, context_dim, and "
                "k_steps must be positive"
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive"
            )

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.k_steps = k_steps
        self.temperature = temperature

        self.gru = nn.GRU(
            latent_dim,
            context_dim,
            batch_first=True,
        )

        self.w_k = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        latent_dim,
                        context_dim,
                    )
                    * 0.05
                )
                for _ in range(k_steps)
            ]
        )

        self.predictor = nn.Linear(
            context_dim,
            latent_dim,
        )

    def encode_sequence(
        self,
        x_seq,
    ):
        if x_seq.ndim != 5:
            raise ValueError(
                "Expected input shaped "
                "(B, T, C, H, W)"
            )

        b, t, c, h, w = x_seq.shape

        z = self.encoder(
            x_seq.reshape(
                b * t,
                c,
                h,
                w,
            )
        )

        z = z.reshape(
            b,
            t,
            self.latent_dim,
        )

        return F.normalize(
            z,
            p=2,
            dim=-1,
            eps=1e-8,
        )

    def predict_next_latent(
        self,
        c_t,
    ):
        """Predict a unit-normalized next latent."""
        return F.normalize(
            self.predictor(c_t),
            p=2,
            dim=-1,
            eps=1e-8,
        )

    def forward(
        self,
        x_seq,
    ):
        z_seq = self.encode_sequence(
            x_seq
        )

        c_seq, _ = self.gru(
            z_seq
        )

        b, t, _ = z_seq.shape

        if b < 2:
            raise ValueError(
                "CPC InfoNCE requires batch_size >= 2"
            )

        cpc_sum = torch.zeros(
            (),
            device=x_seq.device,
        )

        forward_sum = torch.zeros(
            (),
            device=x_seq.device,
        )

        n_contexts = 0
        n_cpc = 0

        for t_idx in range(
            4,
            t - self.k_steps,
        ):
            c_t = c_seq[:, t_idx]

            z_next = z_seq[
                :,
                t_idx + 1,
            ]

            z_pred = self.predict_next_latent(
                c_t
            )

            forward_sum = (
                forward_sum
                + F.mse_loss(
                    z_pred,
                    z_next.detach(),
                )
            )

            n_contexts += 1

            for k in range(
                self.k_steps
            ):
                target = z_seq[
                    :,
                    t_idx + 1 + k,
                ]

                pred = F.normalize(
                    c_t @ self.w_k[k].T,
                    p=2,
                    dim=-1,
                    eps=1e-8,
                )

                logits = (
                    pred @ target.T
                ) / self.temperature

                labels = torch.arange(
                    b,
                    device=x_seq.device,
                )

                cpc_sum = (
                    cpc_sum
                    + F.cross_entropy(
                        logits,
                        labels,
                    )
                )

                n_cpc += 1

        if n_contexts == 0 or n_cpc == 0:
            raise ValueError(
                "Sequence is too short for "
                "configured CPC context"
            )

        return (
            cpc_sum / n_cpc
            + forward_sum / n_contexts
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train temporal CPC on cartpole."
    )

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=12,
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=16,
    )
    p.add_argument(
        "--eval-seq-len",
        type=int,
        default=25,
    )
    p.add_argument(
        "--latent-dim",
        type=int,
        default=16,
    )
    p.add_argument(
        "--context-dim",
        type=int,
        default=32,
    )
    p.add_argument(
        "--k-steps",
        type=int,
        default=3,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--max-train-frames",
        type=int,
        default=4000,
    )
    p.add_argument(
        "--max-val-frames",
        type=int,
        default=1000,
    )
    p.add_argument(
        "--gap-start",
        type=int,
        default=6,
    )
    p.add_argument(
        "--gap-length",
        type=int,
        default=5,
    )
    p.add_argument(
        "--gap-sequences",
        type=int,
        default=32,
        help="Maximum number of evenly spaced validation windows for gap scoring.",
    )
    p.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help="Evaluate existing checkpoints without retraining the CPC model.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default="data",
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default="ch03/figures",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default="ch03/results",
    )
    p.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
    )

    return p.parse_args()


def make_loader(
    dataset,
    batch_size,
    seed=None,
    shuffle=False,
):
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def evenly_spaced_subset(dataset, max_items: int):
    """Select validation windows across the full dataset, not only its start."""
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if len(dataset) == 0:
        raise ValueError("Cannot select from an empty sequence dataset")

    n_items = min(len(dataset), max_items)
    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=n_items,
        dtype=np.int64,
    )
    indices = np.unique(indices)
    return Subset(dataset, indices.tolist())


def validate_checkpoint_evaluation(args) -> dict:
    """Check that cached CPC checkpoints match the original training run."""
    metrics_path = os.path.join(
        args.results_dir,
        "table_03_05_cpc_metrics.json",
    )
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(
            f"Cannot validate existing run: missing {metrics_path}"
        )

    saved_metrics = load_json(metrics_path)
    if saved_metrics.get("seeds") != args.seeds:
        raise ValueError(
            "Requested seeds do not match the saved CPC run: "
            f"{args.seeds} vs {saved_metrics.get('seeds')}"
        )

    saved_config = saved_metrics.get("config", {})
    training_fields = (
        "epochs",
        "batch_size",
        "lr",
        "seq_len",
        "eval_seq_len",
        "latent_dim",
        "context_dim",
        "k_steps",
        "temperature",
        "max_train_frames",
        "max_val_frames",
        "gap_start",
        "gap_length",
    )
    mismatches = {
        field: (saved_config.get(field), getattr(args, field))
        for field in training_fields
        if saved_config.get(field) != getattr(args, field)
    }
    if mismatches:
        raise ValueError(
            "Existing checkpoint training configuration does not match "
            f"the requested evaluation: {mismatches}"
        )

    for seed in args.seeds:
        checkpoint_path = os.path.join(
            args.checkpoints_dir,
            f"temporal_cpc_seed_{seed}.pt",
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Missing CPC checkpoint for seed {seed}: {checkpoint_path}"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        expected = {
            "seed": seed,
            "latent_dim": args.latent_dim,
            "context_dim": args.context_dim,
            "k_steps": args.k_steps,
            "temperature": args.temperature,
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Checkpoint for seed {seed} has incompatible metadata: "
                f"{mismatches}"
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint for seed {seed} has no model_state_dict"
            )

    return saved_metrics


def extract_temporal_features(
    model,
    loader,
    device,
    time_index=8,
):
    z_all = []
    c_all = []
    y_all = []
    next_y_all = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            images = batch[
                "images"
            ].to(device)

            states = (
                batch[
                    "physics_states"
                ]
                .cpu()
                .numpy()
            )

            z_seq = model.encode_sequence(
                images
            )

            c_seq, _ = model.gru(
                z_seq
            )

            z_all.append(
                z_seq[
                    :,
                    time_index,
                ]
                .cpu()
                .numpy()
            )

            c_all.append(
                c_seq[
                    :,
                    time_index,
                ]
                .cpu()
                .numpy()
            )

            y_all.append(
                states[:, time_index]
            )

            next_y_all.append(
                states[
                    :,
                    time_index + 1,
                ]
            )

    if not z_all:
        raise RuntimeError(
            "Temporal feature loader "
            "produced no batches"
        )

    return (
        np.concatenate(
            z_all,
            axis=0,
        ),
        np.concatenate(
            c_all,
            axis=0,
        ),
        np.concatenate(
            y_all,
            axis=0,
        ),
        np.concatenate(
            next_y_all,
            axis=0,
        ),
    )


def fit_state_history_predictor(
    raw_states,
    raw_episode_ids,
):
    """Fit an angle-aware linear state extrapolation baseline."""
    x = []
    y = []

    for i in range(
        1,
        len(raw_states) - 1,
    ):
        if (
            raw_episode_ids[i - 1]
            == raw_episode_ids[i]
            == raw_episode_ids[i + 1]
        ):
            x.append(
                np.concatenate(
                    [
                        raw_states[i - 1],
                        raw_states[i],
                    ]
                )
            )

            y.append(
                raw_states[i + 1]
            )

    if not x:
        raise RuntimeError(
            "No valid consecutive "
            "state triples found"
        )

    return StateLinearProbe().fit(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        np.asarray(
            y,
            dtype=np.float64,
        ),
    )


def rollout_gap(
    model,
    sample_images,
    gap_start,
    gap_end,
    state_probe,
):
    """Roll normalized latent dynamics through the missing interval."""
    model.eval()

    with torch.no_grad():
        z_full = model.encode_sequence(
            sample_images
        )

        _, t, _ = z_full.shape

        if not (
            0 < gap_start < gap_end <= t
        ):
            raise ValueError(
                "Invalid gap interval"
            )

        hidden = None
        c_current = None
        z_used = []

        for step in range(t):
            if (
                step < gap_start
                or step >= gap_end
            ):
                z_in = z_full[
                    :,
                    step : step + 1,
                ]
            else:
                if c_current is None:
                    raise RuntimeError(
                        "Missing context state "
                        "at gap start"
                    )

                z_in = (
                    model.predict_next_latent(
                        c_current
                    )
                    .unsqueeze(1)
                )

            c_out, hidden = model.gru(
                z_in,
                hidden,
            )

            c_current = c_out[:, 0]

            z_used.append(
                z_in[:, 0]
            )

        z_rollout = torch.stack(
            z_used,
            dim=1,
        )

    latent_norms = torch.linalg.vector_norm(
        z_rollout,
        dim=-1,
    )

    if not torch.isfinite(
        latent_norms
    ).all():
        raise FloatingPointError(
            "Non-finite latent during "
            "CPC rollout"
        )

    pred_states = state_probe.predict(
        z_rollout[
            0
        ]
        .cpu()
        .numpy()
    )

    observed_states = state_probe.predict(
        z_full[
            0
        ]
        .cpu()
        .numpy()
    )

    return (
        pred_states,
        observed_states,
        float(
            latent_norms.max().item()
        ),
    )


def linear_extrapolation_gap(
    history_probe,
    true_states,
    gap_start,
    gap_end,
):
    pred = np.full_like(
        true_states,
        np.nan,
        dtype=np.float64,
    )

    history = [
        true_states[
            gap_start - 2
        ].copy(),
        true_states[
            gap_start - 1
        ].copy(),
    ]

    for step in range(
        gap_start,
        gap_end,
    ):
        x = np.concatenate(
            [
                history[-2],
                history[-1],
            ]
        )[None, :]

        next_state = history_probe.predict(
            x
        )[0]

        pred[step] = next_state
        history.append(
            next_state
        )

    return pred


def evaluate_gap(
    model,
    gap_loader,
    state_probe,
    history_probe,
    args,
    device,
):
    gap_true = []
    gap_cpc = []
    gap_linear = []

    first_figure_data = None
    max_latent_norm = 0.0

    gap_end = (
        args.gap_start
        + args.gap_length
    )

    with torch.no_grad():
        for sequence_index, batch in enumerate(
            gap_loader
        ):
            if sequence_index >= args.gap_sequences:
                break

            images = batch[
                "images"
            ].to(device)

            states = (
                batch[
                    "physics_states"
                ]
                .cpu()
                .numpy()
            )

            for sample_index in range(
                len(images)
            ):
                (
                    pred_states,
                    observed_states,
                    sample_max_norm,
                ) = rollout_gap(
                    model,
                    images[
                        sample_index :
                        sample_index + 1
                    ],
                    args.gap_start,
                    gap_end,
                    state_probe,
                )

                max_latent_norm = max(
                    max_latent_norm,
                    sample_max_norm,
                )

                linear = linear_extrapolation_gap(
                    history_probe,
                    states[sample_index],
                    args.gap_start,
                    gap_end,
                )

                gap_true.append(
                    states[
                        sample_index,
                        args.gap_start : gap_end,
                    ]
                )

                gap_cpc.append(
                    pred_states[
                        args.gap_start : gap_end
                    ]
                )

                gap_linear.append(
                    linear[
                        args.gap_start : gap_end
                    ]
                )

                if first_figure_data is None:
                    first_figure_data = {
                        "states": states[
                            sample_index
                        ],
                        "cpc": pred_states,
                        "observed": observed_states,
                        "linear": linear,
                    }

                if (
                    len(gap_true)
                    >= args.gap_sequences
                ):
                    break

            if (
                len(gap_true)
                >= args.gap_sequences
            ):
                break

    if not gap_true:
        raise RuntimeError(
            "No sequences were available "
            "for gap evaluation"
        )

    true = np.concatenate(
        gap_true,
        axis=0,
    )

    cpc = np.concatenate(
        gap_cpc,
        axis=0,
    )

    linear = np.concatenate(
        gap_linear,
        axis=0,
    )

    angular = infer_angular_position_indices(
        true.shape[1]
    )

    r2_cpc = compute_structured_r2_score(
        true,
        cpc,
        angular,
    )

    r2_linear = compute_structured_r2_score(
        true,
        linear,
        angular,
    )

    return (
        {
            "num_sequences": int(
                len(gap_true)
            ),
            "cpc_gap_r2_per_variable": (
                r2_cpc.tolist()
            ),
            "cpc_gap_mean_r2": float(
                np.mean(r2_cpc)
            ),
            "linear_gap_r2_per_variable": (
                r2_linear.tolist()
            ),
            "linear_gap_mean_r2": float(
                np.mean(r2_linear)
            ),
            "max_rollout_latent_norm": float(
                max_latent_norm
            ),
        },
        first_figure_data,
    )


def save_gap_figure(
    data,
    args,
    output_path,
):
    states = data["states"]
    cpc = data["cpc"]
    observed = data["observed"]
    linear = data["linear"]

    gap_end = (
        args.gap_start
        + args.gap_length
    )

    t = np.arange(
        len(states)
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 3.6)
    )

    ax.plot(
        t,
        states[:, 0],
        linestyle="-",
        linewidth=2.0,
        label="Ground Truth",
    )

    cpc_gap = np.full(
        len(states),
        np.nan,
    )

    cpc_gap[
        args.gap_start : gap_end
    ] = cpc[
        args.gap_start : gap_end,
        0,
    ]

    ax.plot(
        t,
        cpc_gap,
        linestyle="--",
        linewidth=1.8,
        label="CPC Latent Rollout",
    )

    linear_gap = np.full(
        len(states),
        np.nan,
    )

    linear_gap[
        args.gap_start : gap_end
    ] = linear[
        args.gap_start : gap_end,
        0,
    ]

    ax.plot(
        t,
        linear_gap,
        linestyle="-.",
        linewidth=1.5,
        label="Privileged State Extrapolation",
    )

    frame_only = observed[
        :,
        0,
    ].copy()

    frame_only[
        args.gap_start : gap_end
    ] = np.nan

    ax.plot(
        t,
        frame_only,
        linestyle=":",
        marker="o",
        markersize=3.5,
        label="Frame-Only Re-entry",
    )

    ax.axvspan(
        args.gap_start,
        gap_end - 1,
        alpha=0.2,
        label="Unobserved Interval",
    )

    ax.set_xlabel(
        "Time Step $t$"
    )

    ax.set_ylabel(
        "Cart Position"
    )

    ax.set_title(
        "Unobserved Interval Test: Five Missing Frames"
    )

    ax.grid(
        True,
        linestyle="--",
        alpha=0.3,
    )

    ax.legend(
        fontsize=8
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.seeds:
        raise ValueError(
            "At least one seed is required"
        )

    if (
        args.epochs <= 0
        or args.batch_size < 2
        or args.lr <= 0
    ):
        raise ValueError(
            "Invalid optimization settings"
        )

    if (
        args.seq_len < 6
        or args.k_steps < 1
        or args.eval_seq_len
        <= args.gap_start + args.gap_length
    ):
        raise ValueError(
            "Invalid sequence or gap settings"
        )

    if (
        args.gap_start < 2
        or args.gap_length <= 0
        or args.gap_sequences <= 0
    ):
        raise ValueError(
            "Invalid gap settings"
        )

    set_seed(args.seeds[0])

    if args.evaluate_checkpoints:
        saved_metrics = validate_checkpoint_evaluation(args)
        print(
            "Evaluation-only mode: validated existing CPC checkpoints; "
            "no model training will be performed."
        )
    else:
        saved_metrics = None

    device = get_device()

    train_raw = load_dataset_npz(
        os.path.join(
            args.data_dir,
            "dmc_cartpole_balance_train.npz",
        )
    )

    val_raw = load_dataset_npz(
        os.path.join(
            args.data_dir,
            "dmc_cartpole_balance_val.npz",
        )
    )

    # Keep the original time series intact. SequenceDataset selects a fixed
    # number of valid windows across episodes without breaking temporal order.
    train_frames = train_raw["frames"]
    train_states = train_raw["physics_states"]
    train_episode_ids = train_raw["episode_ids"]

    val_frames = val_raw["frames"]
    val_states = val_raw["physics_states"]
    val_episode_ids = val_raw["episode_ids"]

    train_ds = SequenceDataset(
        train_frames,
        train_states,
        train_episode_ids,
        args.seq_len,
        max_sequences=args.max_train_frames,
    )

    eval_ds = SequenceDataset(
        val_frames,
        val_states,
        val_episode_ids,
        args.eval_seq_len,
        max_sequences=args.max_val_frames,
    )

    if (
        len(train_ds) == 0
        or len(eval_ds) == 0
    ):
        raise RuntimeError(
            "No valid episode-safe sequences found"
        )

    history_probe = fit_state_history_predictor(
        train_states,
        train_episode_ids,
    )

    angular_indices = (
        infer_angular_position_indices(
            train_states.shape[1]
        )
    )

    gap_eval_ds = evenly_spaced_subset(
        eval_ds,
        args.gap_sequences,
    )
    print(
        f"Gap evaluation will sample {len(gap_eval_ds)} evenly spaced "
        f"windows from {len(eval_ds)} eligible validation windows."
    )

    seed_metrics = []
    first_figure_data = None

    for seed in args.seeds:
        set_seed(seed)

        train_loader = make_loader(
            train_ds,
            args.batch_size,
            seed=seed,
            shuffle=True,
        )

        model = TemporalCPC(
            ConvEncoder(
                latent_dim=args.latent_dim
            ).to(device),
            latent_dim=args.latent_dim,
            context_dim=args.context_dim,
            k_steps=args.k_steps,
            temperature=args.temperature,
        ).to(device)

        if args.evaluate_checkpoints:
            checkpoint_path = os.path.join(
                args.checkpoints_dir,
                f"temporal_cpc_seed_{seed}.pt",
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )
            print(
                f"Seed {seed} | loaded existing checkpoint; "
                "skipping training."
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
            )

            for epoch in range(
                1,
                args.epochs + 1,
            ):
                model.train()

                total = 0.0
                total_examples = 0

                for batch in train_loader:
                    loss = model(
                        batch["images"].to(device)
                    )

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite CPC loss at "
                            f"seed={seed}, epoch={epoch}"
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                    )
                    optimizer.step()

                    n = len(batch["images"])
                    total += float(loss.item()) * n
                    total_examples += n

                if total_examples == 0:
                    raise RuntimeError(
                        "CPC DataLoader produced zero examples"
                    )

                if epoch % 4 == 0 or epoch == args.epochs:
                    print(
                        f"Seed {seed} | "
                        f"Epoch {epoch:02d}/{args.epochs:02d} | "
                        f"Loss {total / total_examples:.4f}"
                    )

        train_z, train_c, train_y, train_next_y = (
            extract_temporal_features(
                model,
                make_loader(
                    train_ds,
                    args.batch_size,
                    shuffle=False,
                ),
                device,
            )
        )

        val_z, val_c, val_y, val_next_y = (
            extract_temporal_features(
                model,
                make_loader(
                    eval_ds,
                    args.batch_size,
                    shuffle=False,
                ),
                device,
            )
        )

        static_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(
            train_z,
            train_y,
        )

        context_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(
            train_c,
            train_y,
        )

        static_next_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(
            train_z,
            train_next_y,
        )

        context_next_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(
            train_c,
            train_next_y,
        )

        static_current_r2, _ = (
            static_probe.score(
                val_z,
                val_y,
            )
        )

        context_current_r2, _ = (
            context_probe.score(
                val_c,
                val_y,
            )
        )

        static_next_r2, _ = (
            static_next_probe.score(
                val_z,
                val_next_y,
            )
        )

        context_next_r2, _ = (
            context_next_probe.score(
                val_c,
                val_next_y,
            )
        )

        gap_stats, figure_data = evaluate_gap(
            model,
            make_loader(
                gap_eval_ds,
                1,
                shuffle=False,
            ),
            static_probe,
            history_probe,
            args,
            device,
        )

        if first_figure_data is None:
            first_figure_data = figure_data

        seed_result = {
            "static_current_position_r2": float(
                static_current_r2[0]
            ),
            "static_current_velocity_r2": float(
                static_current_r2[2]
            ),
            "context_current_position_r2": float(
                context_current_r2[0]
            ),
            "context_current_velocity_r2": float(
                context_current_r2[2]
            ),
            "static_next_state_mean_r2": float(
                np.mean(static_next_r2)
            ),
            "context_next_state_mean_r2": float(
                np.mean(context_next_r2)
            ),
            "static_next_state_position_r2": float(
                np.mean(static_next_r2[:2])
            ),
            "static_next_state_velocity_r2": float(
                np.mean(static_next_r2[2:])
            ),
            "context_next_state_position_r2": float(
                np.mean(context_next_r2[:2])
            ),
            "context_next_state_velocity_r2": float(
                np.mean(context_next_r2[2:])
            ),
            **gap_stats,
        }

        seed_metrics.append(
            seed_result
        )

        os.makedirs(
            args.checkpoints_dir,
            exist_ok=True,
        )

        if not args.evaluate_checkpoints:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "context_dim": args.context_dim,
                    "k_steps": args.k_steps,
                    "temperature": args.temperature,
                    "seed": seed,
                    "experiment_version": EXPERIMENT_VERSION,
                    "latent_normalization": "l2_unit",
                    "gradient_clip_norm": 1.0,
                },
                os.path.join(
                    args.checkpoints_dir,
                    f"temporal_cpc_seed_{seed}.pt",
                ),
            )

        print(
            f"Seed {seed} | "
            f"current static pos="
            f"{static_current_r2[0]:.3f}, "
            f"vel={static_current_r2[2]:.3f} | "
            f"context pos="
            f"{context_current_r2[0]:.3f}, "
            f"vel={context_current_r2[2]:.3f}"
        )

        print(
            f"Seed {seed} | "
            f"next-state static="
            f"{np.mean(static_next_r2):.3f}, "
            f"context="
            f"{np.mean(context_next_r2):.3f} | "
            f"gap CPC="
            f"{gap_stats['cpc_gap_mean_r2']:.3f}, "
            f"linear="
            f"{gap_stats['linear_gap_mean_r2']:.3f} | "
            f"max ||z||="
            f"{gap_stats['max_rollout_latent_norm']:.6f}"
        )

    if first_figure_data is not None:
        os.makedirs(
            args.figures_dir,
            exist_ok=True,
        )

        save_gap_figure(
            first_figure_data,
            args,
            os.path.join(
                args.figures_dir,
                "fig03_16_gap_tracking.png",
            ),
        )

    def mean_std(key):
        values = np.asarray(
            [
                m[key]
                for m in seed_metrics
            ],
            dtype=np.float64,
        )

        return (
            float(values.mean()),
            float(values.std()),
        )

    summary = {
        "static_position_r2": mean_std(
            "static_current_position_r2"
        ),
        "static_velocity_r2": mean_std(
            "static_current_velocity_r2"
        ),
        "context_position_r2": mean_std(
            "context_current_position_r2"
        ),
        "context_velocity_r2": mean_std(
            "context_current_velocity_r2"
        ),
        "static_next_state_mean_r2": mean_std(
            "static_next_state_mean_r2"
        ),
        "context_next_state_mean_r2": mean_std(
            "context_next_state_mean_r2"
        ),
        "cpc_gap_mean_r2": mean_std(
            "cpc_gap_mean_r2"
        ),
        "linear_gap_mean_r2": mean_std(
            "linear_gap_mean_r2"
        ),
    }

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    save_json(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "seeds": args.seeds,
            "config": vars(args),
            "evaluation": {
                "sample_selection": "episode-stratified sequence starts across full split",
                "gap_sampling": "evenly spaced over episode-stratified eligible validation windows",
                "gap_sequence_count": len(gap_eval_ds),
                "eligible_validation_windows": len(eval_ds),
                "checkpoint_only": bool(args.evaluate_checkpoints),
                "prior_metrics_config": (
                    saved_metrics.get("config")
                    if saved_metrics is not None
                    else None
                ),
            },
            "angular_position_indices": list(
                angular_indices
            ),
            "latent_normalization": "l2_unit",
            "gradient_clip_norm": 1.0,
            "summary": summary,
            "per_seed_metrics": seed_metrics,
        },
        os.path.join(
            args.results_dir,
            "table_03_05_cpc_metrics.json",
        ),
    )


if __name__ == "__main__":
    main()