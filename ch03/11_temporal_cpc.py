"""Temporal CPC and missing-observation tests for Chapter 3.

The CPC objective uses a causal GRU context and raw bilinear InfoNCE scores.
Latent representations are not L2-normalized and no temperature is applied.
State probes use standardized features and circular evaluation for angular positions.

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

import numpy as np
import torch
import torch.nn.functional as F
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

EXPERIMENT_VERSION = 9
CPC_OBJECTIVE_CONTRACT = "canonical_cpc_raw_dot_product_infonce_all_contexts_v3"
RETRIEVAL_EVALUATION_CONTRACT_VERSION = 1
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
        "--max-train-sequences", "--max-train-frames",
        dest="max_train_sequences",
        type=int,
        default=4000,
        help="Maximum number of training sequence windows, not individual frames.",
    )
    p.add_argument(
        "--max-val-sequences", "--max-val-frames",
        dest="max_val_sequences",
        type=int,
        default=1000,
        help="Maximum number of validation sequence windows, not individual frames.",
    )
    p.add_argument(
        "--retrieval-sequences", "--gap-sequences",
        dest="retrieval_sequences",
        type=int,
        default=200,
        help="Number of non-overlapping, episode-balanced validation windows used for CPC future-latent retrieval.",
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
    # InfoNCE needs at least two sequences per batch. Drop a trailing singleton
    # batch during training; retrieval evaluation explicitly records skipped
    # singleton batches if a user requests an unusual sample count.
    drop_last = bool(shuffle and len(dataset) > batch_size and len(dataset) % batch_size == 1)
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(0 if seed is None else seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            drop_last=drop_last,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def evaluate_cpc_future_retrieval(model, loader, device) -> dict:
    """Evaluate CPC using its trained InfoNCE compatibility scores.

    For every valid context and each trained future horizon, each row scores
    its own future latent against future latents from other held-out sequences
    in the same batch. No predicted embedding is fed back into the GRU and no
    pointwise latent MSE is used. Higher top-1 retrieval and positive-vs-hardest-
    negative margin, plus lower InfoNCE loss, indicate better objective-aligned
    future identification.
    """
    if int(getattr(model, "k_steps", 0)) < 1:
        raise ValueError("CPC model must expose a positive k_steps value")
    was_training = model.training
    model.eval()
    k_steps = int(model.k_steps)
    totals = {
        horizon: {
            "loss_sum": 0.0,
            "top1_correct": 0,
            "top5_correct": 0,
            "positive_score_sum": 0.0,
            "hardest_negative_score_sum": 0.0,
            "margin_sum": 0.0,
            "chance_sum": 0.0,
            "num_context_examples": 0,
            "candidate_count_sum": 0.0,
            "num_batches": 0,
        }
        for horizon in range(1, k_steps + 1)
    }
    num_sequences = 0
    skipped_singleton_batches = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            z_seq = model.encode_sequence(images)
            c_seq, _ = model.gru(z_seq)
            batch_size, seq_len, _ = z_seq.shape
            num_sequences += int(batch_size)
            if batch_size < 2:
                skipped_singleton_batches += 1
                continue
            if seq_len <= k_steps:
                raise ValueError(
                    f"Sequence length {seq_len} is too short for k_steps={k_steps}"
                )
            labels = torch.arange(batch_size, device=device)
            diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)
            for time_index in range(seq_len - k_steps):
                context = c_seq[:, time_index]
                for horizon in range(1, k_steps + 1):
                    target = z_seq[:, time_index + horizon]
                    scores = (context @ model.w_k[horizon - 1].T) @ target.T
                    loss_sum = F.cross_entropy(scores, labels, reduction="sum")
                    positive = scores.diagonal()
                    hardest_negative = scores.masked_fill(
                        diagonal, -torch.inf
                    ).max(dim=1).values
                    top1 = scores.argmax(dim=1).eq(labels).sum().item()
                    top_k = min(5, batch_size)
                    top_k_indices = scores.topk(top_k, dim=1).indices
                    top5 = top_k_indices.eq(labels[:, None]).any(dim=1).sum().item()
                    item = totals[horizon]
                    item["loss_sum"] += float(loss_sum.item())
                    item["top1_correct"] += int(top1)
                    item["top5_correct"] += int(top5)
                    item["positive_score_sum"] += float(positive.sum().item())
                    item["hardest_negative_score_sum"] += float(hardest_negative.sum().item())
                    item["margin_sum"] += float((positive - hardest_negative).sum().item())
                    item["chance_sum"] += 1.0
                    item["num_context_examples"] += int(batch_size)
                    item["candidate_count_sum"] += float(batch_size * batch_size)
                    item["num_batches"] += 1
    model.train(was_training)

    by_horizon = {}
    for horizon, item in totals.items():
        n = item["num_context_examples"]
        if n == 0:
            raise RuntimeError(
                f"No CPC retrieval examples for horizon={horizon}; "
                "provide at least one validation batch of size >= 2."
            )
        # chance_sum adds one expected correct match per context position and
        # batch, then divides by all candidate rows to report the mean 1/B rate.
        by_horizon[str(horizon)] = {
            "infonce_loss": item["loss_sum"] / n,
            "top1_accuracy": item["top1_correct"] / n,
            "top5_accuracy": item["top5_correct"] / n,
            "random_top1_chance": item["chance_sum"] / n,
            "mean_positive_score": item["positive_score_sum"] / n,
            "mean_hardest_negative_score": item["hardest_negative_score_sum"] / n,
            "mean_positive_minus_hardest_negative": item["margin_sum"] / n,
            "num_context_examples": n,
            "mean_candidate_count": item["candidate_count_sum"] / max(1, item["num_context_examples"]),
        }
    return {
        "by_horizon": by_horizon,
        "num_sequences_seen": num_sequences,
        "skipped_singleton_batches": skipped_singleton_batches,
        "candidate_definition": "future latents from other held-out, episode-safe windows in the same batch",
        "method": "standard CPC bilinear compatibility scores and multi-horizon InfoNCE retrieval",
    }


def summarize_cpc_future_retrieval(seed_metrics: list[dict]) -> dict:
    """Aggregate retrieval results over independent random seeds."""
    if not seed_metrics:
        raise ValueError("seed_metrics cannot be empty")
    horizons = sorted(
        seed_metrics[0]["cpc_future_retrieval"]["by_horizon"],
        key=int,
    )
    metric_names = (
        "infonce_loss",
        "top1_accuracy",
        "top5_accuracy",
        "random_top1_chance",
        "mean_positive_score",
        "mean_hardest_negative_score",
        "mean_positive_minus_hardest_negative",
    )
    summary = {"by_horizon": {}}
    for horizon in horizons:
        summary["by_horizon"][horizon] = {}
        for metric in metric_names:
            values = [
                float(result["cpc_future_retrieval"]["by_horizon"][horizon][metric])
                for result in seed_metrics
            ]
            summary["by_horizon"][horizon][metric] = {
                "mean_over_seeds": float(np.mean(values)),
                "std_over_seeds": float(np.std(values)),
                "per_seed": values,
            }
    summary["interpretation"] = (
        "Top-1 retrieval should be compared with random_top1_chance; "
        "InfoNCE loss and positive-minus-hardest-negative margin are direct "
        "diagnostics of CPC's trained compatibility objective. This is not a "
        "claim that CPC directly regresses future latent vectors."
    )
    return summary


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
            f"Requested {max_items} retrieval windows but selected "
            f"{len(selected_indices)}."
        )

    return Subset(dataset, selected_indices)


def sequence_window_metadata(
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
        raise RuntimeError("Retrieval evaluation selected duplicate window starts")

    counts = {}
    starts_by_episode = {}
    for start, episode in zip(starts, episodes):
        end = start + dataset.seq_len
        if not np.all(dataset.episode_ids[start:end] == episode):
            raise RuntimeError(
                f"Retrieval window [{start}, {end}) crosses an episode boundary"
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
    if saved_metrics.get("experiment_version") not in {8, EXPERIMENT_VERSION}:
        raise ValueError(
            "Checkpoint-only evaluation refused: the metrics file was produced "
            "by an incompatible CPC experiment version."
        )
    if saved_metrics.get("latent_normalization") != "none":
        raise ValueError(
            "Checkpoint-only evaluation refused: the saved run does not use raw "
            "encoder latents."
        )
    if saved_metrics.get("objective_contract") != CPC_OBJECTIVE_CONTRACT:
        raise ValueError(
            "Checkpoint-only evaluation refused: the metrics/checkpoints do not "
            "declare the canonical multi-horizon CPC InfoNCE objective. The prior "
            "run used an auxiliary one-step MSE head; train fresh checkpoints into "
            "a separate results/checkpoints directory before evaluating."
        )
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
    )
    mismatches = {
        field: (saved_config.get(field), getattr(args, field))
        for field in training_fields
        if saved_config.get(field) != getattr(args, field)
    }
    # Version-8 runs called these sequence-window limits "frames". Accept the
    # legacy metadata key only after verifying its value exactly.
    for current_name, legacy_name in (
        ("max_train_sequences", "max_train_frames"),
        ("max_val_sequences", "max_val_frames"),
    ):
        saved_value = saved_config.get(
            current_name, saved_config.get(legacy_name)
        )
        requested_value = getattr(args, current_name)
        if saved_value != requested_value:
            mismatches[current_name] = (saved_value, requested_value)
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
            "latent_normalization": "none",
            "objective_contract": CPC_OBJECTIVE_CONTRACT,
        }
        mismatches = {
            key: (checkpoint.get(key), value)
            for key, value in expected.items()
            if checkpoint.get(key) != value
        }
        if checkpoint.get("experiment_version") not in {8, EXPERIMENT_VERSION}:
            mismatches["experiment_version"] = (
                checkpoint.get("experiment_version"),
                f"one of (8, {EXPERIMENT_VERSION})",
            )
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


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if args.evaluate_checkpoints and args.evaluation_output is not None:
        if os.path.exists(args.evaluation_output):
            raise FileExistsError(
                f"Refusing to overwrite existing evaluation output: {args.evaluation_output}"
            )
    if args.epochs <= 0 or args.batch_size < 2 or args.lr <= 0:
        raise ValueError("epochs and lr must be positive and batch_size must be >= 2")
    if (
        args.seq_len < 10
        or args.k_steps < 1
        or args.k_steps >= args.seq_len
        or args.eval_seq_len < max(10, args.k_steps + 2)
    ):
        raise ValueError(
            "seq_len must be >= 10 for the time-8 linear probe; k_steps must be "
            "smaller than seq_len; eval_seq_len must support the probe and CPC horizons."
        )
    if args.retrieval_sequences < 2:
        raise ValueError("retrieval-sequences must be >= 2")
    if args.max_train_sequences <= 0 or args.max_val_sequences <= 0:
        raise ValueError("max-train-sequences and max-val-sequences must be positive")

    saved_metrics = validate_checkpoint_evaluation(args) if args.evaluate_checkpoints else None
    if args.evaluate_checkpoints:
        print(
            "Evaluation-only mode: loading canonical CPC checkpoints; "
            "training is skipped and standard future-latent retrieval is evaluated."
        )

    set_seed(args.seeds[0])
    device = get_device()
    train_raw = load_dataset_npz(
        os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    )
    val_raw = load_dataset_npz(
        os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")
    )

    train_ds = SequenceDataset(
        train_raw["frames"],
        train_raw["physics_states"],
        train_raw["episode_ids"],
        args.seq_len,
        max_sequences=args.max_train_sequences,
    )
    eval_ds = SequenceDataset(
        val_raw["frames"],
        val_raw["physics_states"],
        val_raw["episode_ids"],
        args.eval_seq_len,
        max_sequences=args.max_val_sequences,
    )
    retrieval_candidates_ds = SequenceDataset(
        val_raw["frames"],
        val_raw["physics_states"],
        val_raw["episode_ids"],
        args.eval_seq_len,
        max_sequences=None,
    )
    if len(train_ds) == 0 or len(eval_ds) == 0 or len(retrieval_candidates_ds) == 0:
        raise RuntimeError("No valid episode-safe sequences found")

    retrieval_eval_ds = episode_stratified_nonoverlapping_subset(
        retrieval_candidates_ds,
        args.retrieval_sequences,
    )
    retrieval_metadata = sequence_window_metadata(
        retrieval_candidates_ds,
        retrieval_eval_ds,
    )
    angular_indices = infer_angular_position_indices(
        train_raw["physics_states"].shape[1]
    )
    print(
        f"CPC retrieval will use {len(retrieval_eval_ds)} non-overlapping, "
        f"episode-safe validation windows from {len(retrieval_candidates_ds)} eligible windows."
    )

    seed_metrics = []
    for seed in args.seeds:
        set_seed(seed)
        train_loader = make_loader(train_ds, args.batch_size, seed=seed, shuffle=True)
        model = TemporalCPC(
            ConvEncoder(latent_dim=args.latent_dim).to(device),
            latent_dim=args.latent_dim,
            context_dim=args.context_dim,
            k_steps=args.k_steps,
        ).to(device)

        if args.evaluate_checkpoints:
            checkpoint_path = os.path.join(
                args.checkpoints_dir, f"temporal_cpc_seed_{seed}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            print(f"Seed {seed} | loaded checkpoint; skipping training.")
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    loss = model(batch["images"].to(device))
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite CPC loss at seed={seed}, epoch={epoch}"
                        )
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += float(loss.detach().item())
                    n_batches += 1
                if n_batches == 0:
                    raise RuntimeError("CPC DataLoader produced zero batches")
                mean_loss = total_loss / n_batches
                if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                    print(
                        f"Seed {seed} | Epoch {epoch:02d}/{args.epochs:02d} | "
                        f"mean InfoNCE loss={mean_loss:.4f}"
                    )

        train_z, train_c, train_y, train_next_y = extract_temporal_features(
            model,
            make_loader(train_ds, args.batch_size, shuffle=False),
            device,
        )
        val_z, val_c, val_y, val_next_y = extract_temporal_features(
            model,
            make_loader(eval_ds, args.batch_size, shuffle=False),
            device,
        )

        static_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(train_z, train_y)
        context_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(train_c, train_y)
        static_next_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(train_z, train_next_y)
        context_next_probe = StateLinearProbe(
            angular_position_indices=angular_indices
        ).fit(train_c, train_next_y)

        static_current_r2, _ = static_probe.score(val_z, val_y)
        context_current_r2, _ = context_probe.score(val_c, val_y)
        static_next_r2, _ = static_next_probe.score(val_z, val_next_y)
        context_next_r2, _ = context_next_probe.score(val_c, val_next_y)

        retrieval_loader = make_loader(
            retrieval_eval_ds,
            args.batch_size,
            seed=seed,
            shuffle=True,
        )
        future_retrieval = evaluate_cpc_future_retrieval(
            model, retrieval_loader, device
        )

        seed_result = {
            "static_current_r2_per_variable": [float(v) for v in static_current_r2],
            "context_current_r2_per_variable": [float(v) for v in context_current_r2],
            "static_next_r2_per_variable": [float(v) for v in static_next_r2],
            "context_next_r2_per_variable": [float(v) for v in context_next_r2],
            "static_current_position_r2": float(static_current_r2[0]),
            "static_current_velocity_r2": float(static_current_r2[2]),
            "context_current_position_r2": float(context_current_r2[0]),
            "context_current_velocity_r2": float(context_current_r2[2]),
            "static_next_state_mean_r2": float(np.mean(static_next_r2)),
            "context_next_state_mean_r2": float(np.mean(context_next_r2)),
            "static_next_state_position_r2": float(np.mean(static_next_r2[:2])),
            "static_next_state_velocity_r2": float(np.mean(static_next_r2[2:])),
            "context_next_state_position_r2": float(np.mean(context_next_r2[:2])),
            "context_next_state_velocity_r2": float(np.mean(context_next_r2[2:])),
            "cpc_future_retrieval": future_retrieval,
        }
        seed_metrics.append(seed_result)

        if not args.evaluate_checkpoints:
            os.makedirs(args.checkpoints_dir, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": args.latent_dim,
                    "context_dim": args.context_dim,
                    "k_steps": args.k_steps,
                    "seed": seed,
                    "experiment_version": EXPERIMENT_VERSION,
                    "objective_contract": CPC_OBJECTIVE_CONTRACT,
                    "sample_selection": "episode_stratified_sequence_starts_across_full_split",
                    "latent_normalization": "none",
                    "gradient_clip_norm": 1.0,
                },
                os.path.join(args.checkpoints_dir, f"temporal_cpc_seed_{seed}.pt"),
            )

        one_step = future_retrieval["by_horizon"]["1"]
        print(
            f"Seed {seed} | current position R2: static={static_current_r2[0]:.3f}, "
            f"context={context_current_r2[0]:.3f}; current velocity R2: "
            f"static={static_current_r2[2]:.3f}, context={context_current_r2[2]:.3f}"
        )
        print(
            f"Seed {seed} | CPC horizon-1 retrieval: top1={one_step['top1_accuracy']:.3f}, "
            f"random chance={one_step['random_top1_chance']:.3f}, "
            f"InfoNCE={one_step['infonce_loss']:.3f}, "
            f"positive-hardest-negative margin="
            f"{one_step['mean_positive_minus_hardest_negative']:.3f}"
        )

    def mean_std(key):
        values = np.asarray([metric[key] for metric in seed_metrics], dtype=np.float64)
        return {"mean": float(values.mean()), "std": float(values.std()), "per_seed": values.tolist()}

    def per_variable_probe_summary(metric_key):
        return {
            name: {
                "mean": float(np.mean([
                    metric[metric_key][index] for metric in seed_metrics
                ])),
                "std": float(np.std([
                    metric[metric_key][index] for metric in seed_metrics
                ])),
                "per_seed": [
                    float(metric[metric_key][index]) for metric in seed_metrics
                ],
            }
            for index, name in enumerate(CARTPOLE_STATE_NAMES)
        }

    summary = {
        "static_position_r2": mean_std("static_current_position_r2"),
        "static_velocity_r2": mean_std("static_current_velocity_r2"),
        "context_position_r2": mean_std("context_current_position_r2"),
        "context_velocity_r2": mean_std("context_current_velocity_r2"),
        "static_next_state_mean_r2": mean_std("static_next_state_mean_r2"),
        "context_next_state_mean_r2": mean_std("context_next_state_mean_r2"),
        "state_probe_r2_by_variable": {
            "static_current": per_variable_probe_summary("static_current_r2_per_variable"),
            "context_current": per_variable_probe_summary("context_current_r2_per_variable"),
            "static_next": per_variable_probe_summary("static_next_r2_per_variable"),
            "context_next": per_variable_probe_summary("context_next_r2_per_variable"),
        },
        "cpc_future_retrieval": summarize_cpc_future_retrieval(seed_metrics),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    result_path = args.evaluation_output or os.path.join(
        args.results_dir, "table_03_05_cpc_metrics.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(result_path)), exist_ok=True)
    save_json(
        {
            "experiment_version": EXPERIMENT_VERSION,
            "objective_contract": CPC_OBJECTIVE_CONTRACT,
            "retrieval_evaluation_contract_version": RETRIEVAL_EVALUATION_CONTRACT_VERSION,
            "seeds": args.seeds,
            "config": vars(args),
            "evaluation": {
                "state_probe_sample_selection": "episode-stratified sequence starts across the validation split",
                "retrieval_sample_selection": (
                    "episode-stratified, non-overlapping, episode-safe windows; "
                    "windows are shuffled into batches so negative candidates do not come from adjacent windows"
                ),
                "retrieval_sequence_count": len(retrieval_eval_ds),
                "eligible_validation_windows": len(retrieval_candidates_ds),
                "retrieval_window_start_indices": retrieval_metadata["start_indices"],
                "retrieval_window_episode_ids": retrieval_metadata["episode_ids"],
                "retrieval_window_episode_counts": retrieval_metadata["episode_counts"],
                "retrieval_windows_overlap": retrieval_metadata["overlapping_windows"],
                "retrieval_candidate_set": (
                    "For each context, compare its correct future latent with future latents "
                    "from other validation sequences in the same batch."
                ),
                "primary_cpc_metrics": [
                    "multi_horizon_infonce_loss",
                    "future_latent_top1_retrieval_accuracy",
                    "future_latent_top5_retrieval_accuracy",
                    "positive_minus_hardest_negative_score",
                ],
                "custom_recursive_latent_rollout": (
                    "Excluded from this experiment because CPC's bilinear output is a "
                    "compatibility score/vector trained by InfoNCE, not a pointwise latent-regression target."
                ),
                "checkpoint_only": bool(args.evaluate_checkpoints),
                "prior_metrics_config": (
                    saved_metrics.get("config") if saved_metrics is not None else None
                ),
            },
            "latent_normalization": "none",
            "sample_selection": "episode_stratified_sequence_starts_across_full_split",
            "gradient_clip_norm": 1.0,
            "summary": summary,
            "per_seed_metrics": seed_metrics,
        },
        result_path,
    )
    print(f"Saved objective-aligned CPC metrics -> {result_path}")


if __name__ == "__main__":
    main()
