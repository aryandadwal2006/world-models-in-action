"""11_temporal_cpc.py - Temporal CPC and honest missing-observation tests.

The CPC objective uses a causal GRU context and bilinear InfoNCE compatibility.
Latent embeddings are L2-normalized so compatibility scores are bounded.

An explicit one-step latent predictor is trained separately with MSE. Its output
is also normalized before being fed back during recursive rollout, preventing
the learned latent dynamics from numerically exploding.

The evaluation distinguishes current-state decoding, next-state prediction,
and prediction through a real unobserved interval.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from worldmodels.data.dmc_data import load_dataset_npz
from worldmodels.eval.probes import LinearProbe, compute_r2_score
from worldmodels.models.encoders import ConvEncoder
from worldmodels.train import get_device, save_json, set_seed


EXPERIMENT_VERSION = 2


class SequenceDataset(Dataset):
    """Yield episode-safe contiguous sequence windows."""

    def __init__(
        self,
        frames,
        physics_states,
        episode_ids,
        seq_len: int,
    ):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")

        if not (
            len(frames)
            == len(physics_states)
            == len(episode_ids)
        ):
            raise ValueError(
                "frames, physics_states, and episode_ids "
                "must have equal lengths"
            )

        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids
        self.seq_len = seq_len

        self.valid_indices = []

        for start in range(0, len(frames) - seq_len + 1):
            window = episode_ids[start : start + seq_len]

            if np.all(window == window[0]):
                self.valid_indices.append(start)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        start = self.valid_indices[index]
        end = start + self.seq_len

        images = (
            torch.from_numpy(self.frames[start:end])
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
    """Causal GRU CPC model with numerically bounded latent dynamics.

    The encoder output is L2-normalized. Consequently:

        ||z_t||_2 = 1

    for every latent representation.

    The explicit predictor also returns a unit-normalized vector. This makes
    recursive rollout stable because the GRU never receives an unbounded latent
    magnitude merely because the predictor is iterated several times.
    """

    def __init__(
        self,
        encoder,
        latent_dim=16,
        context_dim=32,
        k_steps=3,
        temperature=0.5,
    ):
        super().__init__()

        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        if context_dim <= 0:
            raise ValueError("context_dim must be positive")

        if k_steps <= 0:
            raise ValueError("k_steps must be positive")

        if temperature <= 0:
            raise ValueError("temperature must be positive")

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
                    torch.randn(latent_dim, context_dim)
                    * 0.05
                )
                for _ in range(k_steps)
            ]
        )

        self.predictor = nn.Linear(
            context_dim,
            latent_dim,
        )

    def encode_sequence(self, x_seq):
        """Encode a sequence into unit-normalized latent embeddings."""
        if x_seq.ndim != 5:
            raise ValueError(
                "Expected input shaped (B, T, C, H, W)"
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

    def predict_next_latent(self, c_t):
        """Predict a unit-normalized next latent from GRU context."""
        z_pred = self.predictor(c_t)

        return F.normalize(
            z_pred,
            p=2,
            dim=-1,
            eps=1e-8,
        )

    def forward(self, x_seq):
        """Compute CPC InfoNCE plus explicit one-step prediction loss."""
        z_seq = self.encode_sequence(x_seq)

        c_seq, _ = self.gru(z_seq)

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

            z_next = z_seq[:, t_idx + 1]

            z_pred = self.predict_next_latent(c_t)

            # The target is treated as the detached representation produced
            # by the visual encoder. The predictor is trained to match it.
            forward_sum = (
                forward_sum
                + F.mse_loss(
                    z_pred,
                    z_next.detach(),
                )
            )

            n_contexts += 1

            for k in range(self.k_steps):
                target = z_seq[
                    :,
                    t_idx + 1 + k,
                ]

                pred = c_t @ self.w_k[k].T

                # Both target and predicted compatibility vector are
                # normalized, bounding the dot-product range.
                pred = F.normalize(
                    pred,
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

        if n_contexts == 0:
            raise ValueError(
                "Sequence is too short for the configured CPC context"
            )

        if n_cpc == 0:
            raise RuntimeError(
                "CPC produced zero InfoNCE terms"
            )

        return (
            cpc_sum / n_cpc
            + forward_sum / n_contexts
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train temporal CPC on cartpole."
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
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--eval-seq-len",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--context-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--k-steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-train-frames",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--max-val-frames",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--gap-start",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--gap-length",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--gap-sequences",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--figures-dir",
        type=str,
        default="ch03/figures",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="ch03/results",
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="checkpoints",
    )

    return parser.parse_args()


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
            images = batch["images"].to(device)

            states = (
                batch["physics_states"]
                .cpu()
                .numpy()
            )

            z_seq = model.encode_sequence(
                images
            )

            c_seq, _ = model.gru(z_seq)

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
                states[:, time_index + 1]
            )

    if not z_all:
        raise ValueError(
            "Temporal feature loader produced no batches"
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
    """Fit a privileged linear state extrapolation baseline."""
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
            "No valid consecutive state triples found"
        )

    probe = LinearProbe()

    probe.fit(
        np.asarray(
            x,
            dtype=np.float64,
        ),
        np.asarray(
            y,
            dtype=np.float64,
        ),
    )

    return probe


def rollout_gap(
    model,
    sample_images,
    gap_start,
    gap_end,
    state_probe,
):
    """Roll the normalized learned latent dynamics through a gap."""
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
                        "Missing context state at start of rollout gap"
                    )

                z_pred = (
                    model.predict_next_latent(
                        c_current
                    )
                )

                z_in = z_pred.unsqueeze(1)

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

    # The CPC predictor emits normalized latents, so they are in the same
    # representation space used to fit this state probe.
    pred_states = state_probe.predict(
        z_rollout[0]
        .cpu()
        .numpy()
    )

    observed_states = state_probe.predict(
        z_full[0]
        .cpu()
        .numpy()
    )

    return (
        pred_states,
        observed_states,
    )


def linear_extrapolation_gap(
    history_probe,
    true_states,
    gap_start,
    gap_end,
):
    """Recursively extrapolate the privileged physical-state baseline."""
    pred = np.full_like(
        true_states,
        np.nan,
        dtype=np.float64,
    )

    history = [
        true_states[gap_start - 2].copy(),
        true_states[gap_start - 1].copy(),
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
        history.append(next_state)

    return pred


def evaluate_gap(
    model,
    gap_loader,
    state_probe,
    history_probe,
    args,
    device,
):
    """Evaluate recursive latent prediction through a missing interval."""
    gap_true = []
    gap_cpc = []
    gap_linear = []

    first_figure_data = None

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

            images = batch["images"].to(device)

            states = (
                batch["physics_states"]
                .cpu()
                .numpy()
            )

            for sample_index in range(
                len(images)
            ):
                pred_states, observed_states = rollout_gap(
                    model,
                    images[
                        sample_index : sample_index + 1
                    ],
                    args.gap_start,
                    gap_end,
                    state_probe,
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
                        "states": states[sample_index],
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
            "No sequences were available for gap evaluation"
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

    r2_cpc = compute_r2_score(
        true,
        cpc,
    )

    r2_linear = compute_r2_score(
        true,
        linear,
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
        },
        first_figure_data,
    )


def save_gap_figure(
    data,
    args,
    output_path,
):
    """Save Figure 3.16 for the first evaluated sequence."""
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

    frame_only = observed[:, 0].copy()

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

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive"
        )

    if args.batch_size < 2:
        raise ValueError(
            "batch-size must be at least 2"
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be positive"
        )

    if args.latent_dim <= 0:
        raise ValueError(
            "latent-dim must be positive"
        )

    if args.context_dim <= 0:
        raise ValueError(
            "context-dim must be positive"
        )

    if args.k_steps <= 0:
        raise ValueError(
            "k-steps must be positive"
        )

    if args.temperature <= 0:
        raise ValueError(
            "temperature must be positive"
        )

    if args.max_train_frames <= 0:
        raise ValueError(
            "max-train-frames must be positive"
        )

    if args.max_val_frames <= 0:
        raise ValueError(
            "max-val-frames must be positive"
        )

    if args.gap_start < 2:
        raise ValueError(
            "gap-start must be at least 2"
        )

    if args.gap_length <= 0:
        raise ValueError(
            "gap-length must be positive"
        )

    if args.gap_sequences <= 0:
        raise ValueError(
            "gap-sequences must be positive"
        )

    if args.eval_seq_len <= (
        args.gap_start
        + args.gap_length
    ):
        raise ValueError(
            "eval-seq-len is too short for "
            "the configured gap"
        )

    if args.eval_seq_len <= 9:
        raise ValueError(
            "eval-seq-len must be greater than 9"
        )

    set_seed(args.seeds[0])

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

    train_frames = train_raw["frames"][
        : args.max_train_frames
    ]

    train_states = train_raw["physics_states"][
        : args.max_train_frames
    ]

    train_episode_ids = train_raw["episode_ids"][
        : args.max_train_frames
    ]

    val_frames = val_raw["frames"][
        : args.max_val_frames
    ]

    val_states = val_raw["physics_states"][
        : args.max_val_frames
    ]

    val_episode_ids = val_raw["episode_ids"][
        : args.max_val_frames
    ]

    train_ds = SequenceDataset(
        train_frames,
        train_states,
        train_episode_ids,
        args.seq_len,
    )

    eval_ds = SequenceDataset(
        val_frames,
        val_states,
        val_episode_ids,
        args.eval_seq_len,
    )

    if len(train_ds) == 0:
        raise RuntimeError(
            "No valid episode-safe training sequences were found"
        )

    if len(eval_ds) == 0:
        raise RuntimeError(
            "No valid episode-safe evaluation sequences were found"
        )

    history_probe = fit_state_history_predictor(
        train_states,
        train_episode_ids,
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

        eval_loader = make_loader(
            eval_ds,
            args.batch_size,
            shuffle=False,
        )

        encoder = ConvEncoder(
            latent_dim=args.latent_dim
        ).to(device)

        model = TemporalCPC(
            encoder,
            latent_dim=args.latent_dim,
            context_dim=args.context_dim,
            k_steps=args.k_steps,
            temperature=args.temperature,
        ).to(device)

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
                images = batch[
                    "images"
                ].to(device)

                loss = model(images)

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

                batch_size = len(images)

                total += (
                    float(loss.item())
                    * batch_size
                )

                total_examples += batch_size

            if total_examples == 0:
                raise RuntimeError(
                    "CPC DataLoader produced zero examples"
                )

            average_loss = (
                total
                / total_examples
            )

            if epoch % 4 == 0 or epoch == args.epochs:
                print(
                    f"Seed {seed} | "
                    f"Epoch {epoch:02d}/{args.epochs:02d} | "
                    f"Loss {average_loss:.4f}"
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

        static_probe = LinearProbe().fit(
            train_z,
            train_y,
        )

        context_probe = LinearProbe().fit(
            train_c,
            train_y,
        )

        static_next_probe = LinearProbe().fit(
            train_z,
            train_next_y,
        )

        context_next_probe = LinearProbe().fit(
            train_c,
            train_next_y,
        )

        static_current_r2, _ = static_probe.score(
            val_z,
            val_y,
        )

        context_current_r2, _ = context_probe.score(
            val_c,
            val_y,
        )

        static_next_r2, _ = static_next_probe.score(
            val_z,
            val_next_y,
        )

        context_next_r2, _ = context_next_probe.score(
            val_c,
            val_next_y,
        )

        gap_stats, figure_data = evaluate_gap(
            model,
            make_loader(
                eval_ds,
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
                "rollout_gradient_clip": 1.0,
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
            f"{gap_stats['linear_gap_mean_r2']:.3f}"
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
            "latent_normalization": "l2_unit",
            "rollout_gradient_clip": 1.0,
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