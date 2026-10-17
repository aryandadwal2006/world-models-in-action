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
from torch.utils.data import DataLoader, Subset

from worldmodels.data.dmc_data import (
    load_dataset_npz,
    select_episode_stratified_indices,
)
from worldmodels.eval.probes import (
    StateLinearProbe,
    compute_structured_r2_score,
    infer_angular_position_indices,
)
from worldmodels.data.sequences import SequenceDataset
from worldmodels.models.encoders import ConvEncoder
from worldmodels.models.temporal_cpc import TemporalCPC
from worldmodels.train import (
    get_device,
    load_json,
    save_json,
    set_seed,
)

EXPERIMENT_VERSION = 5
GAP_EVALUATION_CONTRACT_VERSION = 2
CARTPOLE_STATE_NAMES = (
    "cart_position",
    "pole_angle",
    "cart_velocity",
    "pole_angular_velocity",
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
        help=(
            "Number of episode-balanced, non-overlapping validation windows "
            "for gap scoring."
        ),
    )
    p.add_argument(
        "--evaluate-checkpoints",
        action="store_true",
        help="Evaluate existing checkpoints without retraining the CPC model.",
    )
    p.add_argument(
        "--evaluation-output",
        type=str,
        default=None,
        help=(
            "Optional new JSON path for checkpoint-only evaluation. "
            "The existing committed metrics file is used as the checkpoint "
            "configuration reference and is never overwritten."
        ),
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


def episode_stratified_nonoverlapping_subset(
    dataset: SequenceDataset,
    max_items: int,
) -> Subset:
    """Select episode-balanced validation windows without temporal overlap.

    Candidate starts are chosen greedily so consecutive selected windows
    within an episode are separated by at least seq_len frames. Selection is
    then balanced across episode IDs by the shared data utility.
    """
    if max_items <= 0:
        raise ValueError("max_items must be positive")
    if len(dataset) == 0:
        raise ValueError("Cannot select from an empty sequence dataset")

    candidate_starts = []
    start_to_dataset_index = {}
    last_start_by_episode = {}

    for dataset_index, raw_start in enumerate(dataset.valid_indices):
        start = int(raw_start)
        episode = int(dataset.episode_ids[start])
        last_start = last_start_by_episode.get(episode)

        if last_start is None or start - last_start >= dataset.seq_len:
            candidate_starts.append(start)
            start_to_dataset_index[start] = dataset_index
            last_start_by_episode[episode] = start

    if len(candidate_starts) < max_items:
        raise ValueError(
            f"Requested {max_items} non-overlapping episode-safe windows, "
            f"but only {len(candidate_starts)} are available."
        )

    selected_starts = select_episode_stratified_indices(
        dataset.episode_ids,
        max_items,
        candidate_indices=np.asarray(candidate_starts, dtype=np.int64),
    )
    selected_indices = [
        start_to_dataset_index[int(start)]
        for start in selected_starts
    ]

    if len(selected_indices) != max_items:
        raise RuntimeError(
            f"Requested {max_items} gap windows but selected "
            f"{len(selected_indices)}."
        )

    return Subset(dataset, selected_indices)


def gap_window_metadata(
    dataset: SequenceDataset,
    subset: Subset,
) -> dict:
    """Describe and validate the selected gap windows for reproducibility."""
    starts = [
        int(dataset.valid_indices[int(index)])
        for index in subset.indices
    ]
    episodes = [
        int(dataset.episode_ids[start])
        for start in starts
    ]

    if len(starts) != len(set(starts)):
        raise RuntimeError("Gap evaluation selected duplicate window starts")

    counts = {}
    starts_by_episode = {}
    for start, episode in zip(starts, episodes):
        end = start + dataset.seq_len
        if not np.all(dataset.episode_ids[start:end] == episode):
            raise RuntimeError(
                f"Gap window [{start}, {end}) crosses an episode boundary"
            )
        counts[str(episode)] = counts.get(str(episode), 0) + 1
        starts_by_episode.setdefault(episode, []).append(start)

    for episode, episode_starts in starts_by_episode.items():
        episode_starts.sort()
        if any(
            right - left < dataset.seq_len
            for left, right in zip(
                episode_starts,
                episode_starts[1:],
            )
        ):
            raise RuntimeError(
                f"Gap windows overlap in episode {episode}"
            )

    return {
        "start_indices": starts,
        "episode_ids": episodes,
        "episode_counts": dict(sorted(counts.items())),
        "overlapping_windows": False,
    }


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


def _latent_error_metrics(predicted, target) -> dict:
    """Return latent MSE and cosine agreement for matched two-dimensional arrays."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must share shape (N, D)")
    if predicted.shape[0] == 0:
        raise ValueError("latent metric inputs cannot be empty")
    mse = float(np.mean((predicted - target) ** 2))
    pred_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    denom = np.maximum(pred_norm * target_norm, 1e-12)
    cosine = np.sum(predicted * target, axis=1) / denom
    return {
        "mse": mse,
        "mean_cosine_similarity": float(np.mean(cosine)),
        "num_predictions": int(predicted.shape[0]),
    }


def evaluate_latent_prediction_diagnostics(
    model,
    loader,
    device,
    rollout_start: int,
    max_horizon: int = 5,
) -> dict:
    """Contrast one-step predictions with free autoregressive latent rollouts.

    Teacher-forced predictions use true encoded frames to form every context.
    Rollout predictions feed each predicted latent back into the GRU.
    """
    if rollout_start < 4 or max_horizon < 1:
        raise ValueError("rollout_start must be >= 4 and max_horizon >= 1")

    teacher_predictions = []
    teacher_targets = []
    rollout_predictions = {horizon: [] for horizon in range(1, max_horizon + 1)}
    rollout_targets = {horizon: [] for horizon in range(1, max_horizon + 1)}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            z_seq = model.encode_sequence(images)
            batch_size, seq_len, _ = z_seq.shape
            if rollout_start + max_horizon >= seq_len:
                raise ValueError(
                    "Requested rollout exceeds sequence length: "
                    f"start={rollout_start}, horizon={max_horizon}, length={seq_len}"
                )

            # One-step forecasts use true encoded observations at every context.
            contexts, _ = model.gru(z_seq)
            k_steps = int(getattr(model, "k_steps", 0))
            if k_steps < 1:
                raise ValueError("Model must expose a positive k_steps training horizon")
            # Match the contexts on which the one-step auxiliary head was trained.
            for time_index in range(4, seq_len - k_steps):
                predicted = model.predict_next_latent(contexts[:, time_index])
                teacher_predictions.append(predicted.cpu().numpy())
                teacher_targets.append(z_seq[:, time_index + 1].cpu().numpy())

            # Open-loop forecasts start after the final observed frame.
            prefix, hidden = model.gru(z_seq[:, : rollout_start + 1])
            context = prefix[:, -1]
            for horizon in range(1, max_horizon + 1):
                predicted = model.predict_next_latent(context)
                target = z_seq[:, rollout_start + horizon]
                rollout_predictions[horizon].append(predicted.cpu().numpy())
                rollout_targets[horizon].append(target.cpu().numpy())
                context_output, hidden = model.gru(
                    predicted.unsqueeze(1),
                    hidden,
                )
                context = context_output[:, 0]

    if not teacher_predictions:
        raise RuntimeError("No teacher-forced latent predictions were evaluated")
    teacher = _latent_error_metrics(
        np.concatenate(teacher_predictions, axis=0),
        np.concatenate(teacher_targets, axis=0),
    )
    rollout = {}
    for horizon in range(1, max_horizon + 1):
        if not rollout_predictions[horizon]:
            raise RuntimeError(f"No rollout predictions for horizon {horizon}")
        rollout[str(horizon)] = _latent_error_metrics(
            np.concatenate(rollout_predictions[horizon], axis=0),
            np.concatenate(rollout_targets[horizon], axis=0),
        )
    return {
        "teacher_forced_one_step": teacher,
        "autoregressive_rollout_by_horizon": rollout,
        "rollout_start_index": int(rollout_start),
        "max_horizon": int(max_horizon),
        "target_space": "unit-normalized encoder latents",
    }


def summarize_latent_prediction_diagnostics(seed_metrics: list[dict]) -> dict:
    """Aggregate per-seed forecast diagnostics without treating windows as seeds."""
    if not seed_metrics:
        raise ValueError("seed_metrics cannot be empty")
    teacher = [
        item["latent_prediction_diagnostics"]["teacher_forced_one_step"]
        for item in seed_metrics
    ]
    summary = {"teacher_forced_one_step": {}}
    for metric in ("mse", "mean_cosine_similarity"):
        values = [float(item[metric]) for item in teacher]
        summary["teacher_forced_one_step"][metric] = {
            "mean_over_seeds": float(np.mean(values)),
            "std_over_seeds": float(np.std(values)),
            "per_seed": values,
        }

    max_horizon = max(
        int(horizon)
        for horizon in seed_metrics[0]["latent_prediction_diagnostics"][
            "autoregressive_rollout_by_horizon"
        ]
    )
    summary["autoregressive_rollout_by_horizon"] = {}
    for horizon in range(1, max_horizon + 1):
        summary["autoregressive_rollout_by_horizon"][str(horizon)] = {}
        for metric in ("mse", "mean_cosine_similarity"):
            values = [
                float(
                    item["latent_prediction_diagnostics"][
                        "autoregressive_rollout_by_horizon"
                    ][str(horizon)][metric]
                )
                for item in seed_metrics
            ]
            summary["autoregressive_rollout_by_horizon"][str(horizon)][metric] = {
                "mean_over_seeds": float(np.mean(values)),
                "std_over_seeds": float(np.std(values)),
                "per_seed": values,
            }
    summary["interpretation"] = (
        "Teacher-forced errors use true encoded observations at every context. "
        "Autoregressive errors feed predicted latents back into the GRU. "
        "These are latent-space diagnostics, not physical-state R2 scores."
    )
    return summary


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
    gap_episode_ids,
):
    gap_true = []
    gap_cpc = []
    gap_linear = []
    per_episode_arrays = {}

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

                gap_true_window = states[
                    sample_index,
                    args.gap_start : gap_end,
                ]
                gap_cpc_window = pred_states[
                    args.gap_start : gap_end
                ]
                gap_linear_window = linear[
                    args.gap_start : gap_end
                ]

                window_index = len(gap_true)
                if window_index >= len(gap_episode_ids):
                    raise RuntimeError(
                        "Gap-window episode metadata is shorter than "
                        "the evaluated window set"
                    )
                episode_id = int(gap_episode_ids[window_index])
                episode_arrays = per_episode_arrays.setdefault(
                    episode_id,
                    {"true": [], "cpc": [], "linear": []},
                )
                episode_arrays["true"].append(gap_true_window)
                episode_arrays["cpc"].append(gap_cpc_window)
                episode_arrays["linear"].append(gap_linear_window)

                gap_true.append(gap_true_window)
                gap_cpc.append(gap_cpc_window)
                gap_linear.append(gap_linear_window)

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

    per_episode_metrics = {}
    for episode_id, arrays in sorted(per_episode_arrays.items()):
        episode_true = np.concatenate(arrays["true"], axis=0)
        episode_cpc = np.concatenate(arrays["cpc"], axis=0)
        episode_linear = np.concatenate(arrays["linear"], axis=0)
        episode_cpc_r2 = compute_structured_r2_score(
            episode_true,
            episode_cpc,
            angular,
        )
        episode_linear_r2 = compute_structured_r2_score(
            episode_true,
            episode_linear,
            angular,
        )
        per_episode_metrics[str(episode_id)] = {
            "num_sequences": len(arrays["true"]),
            "cpc_gap_r2_per_variable": episode_cpc_r2.tolist(),
            "cpc_gap_mean_r2": float(np.mean(episode_cpc_r2)),
            "linear_gap_r2_per_variable": episode_linear_r2.tolist(),
            "linear_gap_mean_r2": float(np.mean(episode_linear_r2)),
        }

    return (
        {
            "num_sequences": int(
                len(gap_true)
            ),
            "num_episodes": len(per_episode_metrics),
            "per_episode_metrics": per_episode_metrics,
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
        f"Illustrative gap trace: one validation window, "
        f"{args.gap_length} missing frames"
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

    if args.evaluation_output is not None:
        if not args.evaluate_checkpoints:
            raise ValueError(
                "--evaluation-output is allowed only with "
                "--evaluate-checkpoints so training results are not "
                "accidentally redirected."
            )
        if os.path.exists(args.evaluation_output):
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation output: "
                f"{args.evaluation_output}"
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

    # Latent forecast diagnostics use validation windows matching the training
    # sequence length, so teacher-forced contexts stay within the trained horizon.
    prediction_diag_ds = SequenceDataset(
        val_frames,
        val_states,
        val_episode_ids,
        args.seq_len,
        max_sequences=args.max_val_frames,
    )

    # Gap evaluation has its own complete candidate set. Do not cap it at the
    # 1,000 windows used for the state probes: the gap metric needs enough
    # non-overlapping held-out windows to be stable and reproducible.
    gap_candidates_ds = SequenceDataset(
        val_frames,
        val_states,
        val_episode_ids,
        args.eval_seq_len,
        max_sequences=None,
    )

    if (
        len(train_ds) == 0
        or len(eval_ds) == 0
        or len(gap_candidates_ds) == 0
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

    gap_eval_ds = episode_stratified_nonoverlapping_subset(
        gap_candidates_ds,
        args.gap_sequences,
    )
    gap_metadata = gap_window_metadata(
        gap_candidates_ds,
        gap_eval_ds,
    )
    print(
        f"Gap evaluation will score {len(gap_eval_ds)} non-overlapping "
        f"episode-safe windows from {len(gap_candidates_ds)} eligible "
        f"validation windows across {len(gap_metadata['episode_counts'])} "
        "episodes."
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

        latent_prediction_diagnostics = evaluate_latent_prediction_diagnostics(
            model,
            make_loader(
                prediction_diag_ds,
                args.batch_size,
                shuffle=False,
            ),
            device,
            rollout_start=args.gap_start - 1,
            max_horizon=args.gap_length,
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
            gap_metadata["episode_ids"],
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
            "latent_prediction_diagnostics": latent_prediction_diagnostics,
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
                    "sample_selection": "episode_stratified_sequence_starts_across_full_split",
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

    def per_variable_summary(metric_key):
        return {
            name: {
                "mean": float(
                    np.mean(
                        [
                            metric[metric_key][index]
                            for metric in seed_metrics
                        ]
                    )
                ),
                "std": float(
                    np.std(
                        [
                            metric[metric_key][index]
                            for metric in seed_metrics
                        ]
                    )
                ),
            }
            for index, name in enumerate(CARTPOLE_STATE_NAMES)
        }

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
        "cpc_gap_r2_per_variable": per_variable_summary(
            "cpc_gap_r2_per_variable"
        ),
        "linear_gap_r2_per_variable": per_variable_summary(
            "linear_gap_r2_per_variable"
        ),
        "latent_prediction_diagnostics": summarize_latent_prediction_diagnostics(
            seed_metrics
        ),
    }

    os.makedirs(
        args.results_dir,
        exist_ok=True,
    )

    result_path = (
        args.evaluation_output
        or os.path.join(
            args.results_dir,
            "table_03_05_cpc_metrics.json",
        )
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(result_path)),
        exist_ok=True,
    )

    save_json(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "gap_evaluation_contract_version": GAP_EVALUATION_CONTRACT_VERSION,
            "seeds": args.seeds,
            "config": vars(args),
            "evaluation": {
                "sample_selection": "episode-stratified sequence starts across full split",
                "gap_sampling": (
                    "episode-stratified non-overlapping contiguous windows "
                    "selected from all eligible validation starts"
                ),
                "gap_sequence_count": len(gap_eval_ds),
                "eligible_validation_windows": len(gap_candidates_ds),
                "probe_validation_windows": len(eval_ds),
                "gap_window_start_indices": gap_metadata["start_indices"],
                "gap_window_episode_ids": gap_metadata["episode_ids"],
                "gap_window_episode_counts": gap_metadata["episode_counts"],
                "gap_windows_overlap": gap_metadata["overlapping_windows"],
                "gap_baseline_scope": (
                    "One state-history model is fitted on the training split "
                    "and shared across CPC seeds. Its zero across-seed "
                    "standard deviation is not an independent seed estimate."
                ),
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
            "sample_selection": "episode_stratified_sequence_starts_across_full_split",
            "gradient_clip_norm": 1.0,
            "summary": summary,
            "per_seed_metrics": seed_metrics,
        },
        result_path,
    )
    print(f"Saved CPC metrics -> {result_path}")


if __name__ == "__main__":
    main()