"""11_temporal_cpc.py - Temporal Contrastive Predictive Coding (CPC) and trajectory tests.

Bridges representation learning to predictive world modeling:
1. Encodes sequence frames into representations z_t via ConvEncoder.
2. Aggregates past history into context c_t via causal recurrent model (GRU).
3. Optimizes future latent compatibility f(z_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t) via InfoNCE (Eq 3.20).
4. Evaluates two temporal tests:
   - Unobserved interval tracking: Predicts through dropped-frame gap (Figure 3.16).
   - Multi-step forward state prediction (Table 3.5).
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


class SequenceDataset(Dataset):
    """Dataset yielding contiguous trajectory slices for temporal CPC."""

    def __init__(
        self,
        frames: np.ndarray,
        physics_states: np.ndarray,
        episode_ids: np.ndarray,
        seq_len: int = 16,
    ) -> None:
        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids
        self.seq_len = seq_len

        # Gather all valid starting indices within the same episode
        self.valid_indices = []
        for i in range(len(frames) - seq_len):
            if episode_ids[i] == episode_ids[i + seq_len - 1]:
                self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict:
        start = self.valid_indices[idx]
        end = start + self.seq_len

        # (T, H, W, C) -> (T, C, H, W) float32 in [0, 1]
        seq_frames = torch.from_numpy(self.frames[start:end]).permute(0, 3, 1, 2).float() / 255.0
        seq_phys = torch.from_numpy(self.physics_states[start:end]).float()

        return {
            "images": seq_frames,
            "physics_states": seq_phys,
        }


class TemporalCPC(nn.Module):
    """Contrastive Predictive Coding architecture for visual sequences."""

    def __init__(
        self,
        encoder: ConvEncoder,
        latent_dim: int = 16,
        context_dim: int = 32,
        k_steps: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.k_steps = k_steps

        self.gru = nn.GRU(latent_dim, context_dim, batch_first=True)
        # Bilinear compatibility matrices W_k for k in {1..k_steps}
        self.w_k = nn.ParameterList([
            nn.Parameter(torch.randn(latent_dim, context_dim) * 0.05)
            for _ in range(k_steps)
        ])

    def encode_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Encodes (B, T, C, H, W) images into latent sequences (B, T, D)."""
        b, t, c, h, w = x_seq.shape
        x_flat = x_seq.view(b * t, c, h, w)
        z_flat = self.encoder(x_flat)
        return z_flat.view(b, t, self.latent_dim)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Computes multi-step InfoNCE loss across future steps."""
        z_seq = self.encode_sequence(x_seq)  # (B, T, D)
        b, t, d = z_seq.shape

        # Run GRU over history
        c_seq, _ = self.gru(z_seq)  # (B, T, C_dim)

        loss = torch.tensor(0.0, device=x_seq.device)
        count = 0

        # For each time step t_idx that has k_steps future steps available
        for t_idx in range(4, t - self.k_steps):
            c_t = c_seq[:, t_idx]  # (B, C_dim)

            for k in range(self.k_steps):
                z_target = z_seq[:, t_idx + 1 + k]  # (B, D)
                W = self.w_k[k]  # (D, C_dim)

                # Compatibility logits: f(z_{t+k}, c_t) = z^T W c
                # Predicted embedding from context: W @ c_t -> (B, D)
                pred_z = torch.matmul(c_t, W.T)  # (B, D)

                # Cosine or dot product similarity matrix between all batch items
                logits = torch.matmul(pred_z, z_target.T)  # (B, B)
                labels = torch.arange(b, device=x_seq.device)

                loss += F.cross_entropy(logits, labels)
                count += 1

        return loss / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal CPC training and gap tracking.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Random seeds.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence window length.")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dimensionality.")
    parser.add_argument("--context-dim", type=int, default=32, help="Context GRU dimension.")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory.")
    parser.add_argument("--figures-dir", type=str, default="ch03/figures", help="Figures directory.")
    parser.add_argument("--results-dir", type=str, default="ch03/results", help="Results directory.")
    args = parser.parse_args()

    device = get_device()
    train_file = os.path.join(args.data_dir, "dmc_cartpole_balance_train.npz")
    val_file = os.path.join(args.data_dir, "dmc_cartpole_balance_val.npz")

    train_raw = load_dataset_npz(train_file)
    val_raw = load_dataset_npz(val_file)

    train_ds = SequenceDataset(
        frames=train_raw["frames"][:4000],
        physics_states=train_raw["physics_states"][:4000],
        episode_ids=train_raw["episode_ids"][:4000],
        seq_len=args.seq_len,
    )
    val_ds = SequenceDataset(
        frames=val_raw["frames"][:1000],
        physics_states=val_raw["physics_states"][:1000],
        episode_ids=val_raw["episode_ids"][:1000],
        seq_len=args.seq_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    seed_table_metrics = []

    print(f"=== Training Temporal CPC across seeds {args.seeds} ===")

    for seed in args.seeds:
        set_seed(seed)
        encoder = ConvEncoder(in_channels=3, latent_dim=args.latent_dim).to(device)
        cpc = TemporalCPC(
            encoder=encoder,
            latent_dim=args.latent_dim,
            context_dim=args.context_dim,
            k_steps=3,
        ).to(device)

        optimizer = torch.optim.Adam(cpc.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            cpc.train()
            total_loss = 0.0
            for batch in train_loader:
                x_seq = batch["images"].to(device)
                loss = cpc(x_seq)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(x_seq)

            avg_loss = total_loss / len(train_ds)
            if epoch % 4 == 0 or epoch == args.epochs:
                print(f"Seed {seed} | Epoch {epoch:02d}/{args.epochs:02d} | CPC InfoNCE Loss: {avg_loss:.4f}")

        # Evaluate Probes:
        # 1. State from Context c_t vs State from Static Frame z_t
        cpc.eval()
        train_z, train_c, train_y, train_next_y = [], [], [], []
        val_z, val_c, val_y, val_next_y = [], [], [], []

        with torch.no_grad():
            for batch in train_loader:
                imgs = batch["images"].to(device)
                phys = batch["physics_states"].numpy()
                z_s = cpc.encode_sequence(imgs)
                c_s, _ = cpc.gru(z_s)
                # middle step index 8
                train_z.append(z_s[:, 8].cpu().numpy())
                train_c.append(c_s[:, 8].cpu().numpy())
                train_y.append(phys[:, 8])
                train_next_y.append(phys[:, 9])

            for batch in val_loader:
                imgs = batch["images"].to(device)
                phys = batch["physics_states"].numpy()
                z_s = cpc.encode_sequence(imgs)
                c_s, _ = cpc.gru(z_s)
                val_z.append(z_s[:, 8].cpu().numpy())
                val_c.append(c_s[:, 8].cpu().numpy())
                val_y.append(phys[:, 8])
                val_next_y.append(phys[:, 9])

        tr_z, val_z = np.concatenate(train_z, 0), np.concatenate(val_z, 0)
        tr_c, val_c = np.concatenate(train_c, 0), np.concatenate(val_c, 0)
        tr_y, val_y = np.concatenate(train_y, 0), np.concatenate(val_y, 0)
        tr_ny, val_ny = np.concatenate(train_next_y, 0), np.concatenate(val_next_y, 0)

        # Probe on static frame z_t
        probe_static = LinearProbe().fit(tr_z, tr_y)
        r2_static, mean_static = probe_static.score(val_z, val_y)

        # Probe on context c_t (should capture velocity and history!)
        probe_context = LinearProbe().fit(tr_c, tr_y)
        r2_context, mean_context = probe_context.score(val_c, val_y)

        # Forward prediction test: predict next state s_{t+1} from c_t
        fwd_probe = LinearProbe().fit(tr_c, tr_ny)
        r2_fwd, mean_fwd = fwd_probe.score(val_c, val_ny)

        seed_table_metrics.append({
            "pos_static": float(r2_static[0]),
            "vel_static": float(r2_static[2]),
            "pos_context": float(r2_context[0]),
            "vel_context": float(r2_context[2]),
            "forward_pred_r2": float(mean_fwd),
        })

        print(f"Seed {seed} | Static pos R^2: {r2_static[0]:.3f}, vel R^2: {r2_static[2]:.3f}")
        print(f"Seed {seed} | Context pos R^2: {r2_context[0]:.3f}, vel R^2: {r2_context[2]:.3f}")
        print(f"Seed {seed} | Next-state Forward Pred R^2: {mean_fwd:.3f}")

        # Visual Demonstration for Figure 3.16: Unobserved interval test
        if seed == args.seeds[0]:
            # Pick a contiguous test sequence of 25 frames
            sample_seq = val_ds[0]["images"].unsqueeze(0).to(device)  # (1, T, C, H, W)
            sample_phys = val_ds[0]["physics_states"].numpy()  # (T, 4)
            T = sample_seq.shape[1]

            # Let gap be from step 6 to 11 (5 dropped frames)
            gap_start, gap_end = 6, 11
            with torch.no_grad():
                z_full = cpc.encode_sequence(sample_seq)  # (1, T, D)
                c_full, _ = cpc.gru(z_full)

                # Context-guided latent rollout across the gap:
                # Up to gap_start, feed true z; inside gap, autoregressively feed predicted z:
                # W_0 map: z_pred = c @ W_0
                W0 = cpc.w_k[0]
                c_rollout = []
                z_rollout = []
                h = None
                for step in range(T):
                    if step < gap_start or step >= gap_end:
                        z_in = z_full[:, step : step + 1]
                    else:
                        # Unobserved: advance using context prediction
                        z_pred = torch.matmul(c_rollout[-1], W0.T).unsqueeze(1)
                        z_in = z_pred

                    c_out, h = cpc.gru(z_in, h)
                    c_rollout.append(c_out[:, 0])
                    z_rollout.append(z_in[:, 0].cpu().numpy())

            # Decode cart position using fitted probe
            z_seq_np = np.concatenate(z_rollout, axis=0)  # (T, D)
            pred_gap_states = probe_static.predict(z_seq_np)  # (T, 4)

            # Frame-only re-entry baseline (set to 0 during gap)
            frame_only_states = pred_gap_states.copy()
            frame_only_states[gap_start:gap_end] = np.nan

            # True cart position
            true_pos = sample_phys[:, 0]
            pred_gap_pos = pred_gap_states[:, 0]

            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            t_axis = np.arange(T)

            ax.plot(t_axis, true_pos, color="black", linestyle="-", linewidth=2.0, label="Ground Truth Position")
            ax.plot(t_axis, pred_gap_pos, color="black", linestyle="--", linewidth=1.8, label="CPC Context Prediction")
            ax.plot(t_axis, frame_only_states[:, 0], color="black", linestyle=":", marker="o", markersize=4, label="Frame-Only Re-entry")

            # Highlight dropped frame interval
            ax.axvspan(gap_start, gap_end - 1, color="gray", alpha=0.25, label="Unobserved Interval (Dropped Frames)")

            ax.set_title("Unobserved Interval Test: Tracking Through Missing Observations", fontsize=10, pad=8)
            ax.set_xlabel("Time Step $t$", fontsize=9)
            ax.set_ylabel("Cart Position", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(frameon=True, fontsize=8)

            os.makedirs(args.figures_dir, exist_ok=True)
            fig_path = os.path.join(args.figures_dir, "fig03_16_gap_tracking.png")
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved Figure 3.16 -> {fig_path}")

    # Output Table 3.5
    print("\n" + "=" * 70)
    print("Table 3.5 Static vs Temporal Representations (Mean ± Std over 3 seeds)")
    print("=" * 70)
    p_stat = np.mean([m["pos_static"] for m in seed_table_metrics])
    p_stat_s = np.std([m["pos_static"] for m in seed_table_metrics])
    v_stat = np.mean([m["vel_static"] for m in seed_table_metrics])
    v_stat_s = np.std([m["vel_static"] for m in seed_table_metrics])

    p_ctx = np.mean([m["pos_context"] for m in seed_table_metrics])
    p_ctx_s = np.std([m["pos_context"] for m in seed_table_metrics])
    v_ctx = np.mean([m["vel_context"] for m in seed_table_metrics])
    v_ctx_s = np.std([m["vel_context"] for m in seed_table_metrics])

    fwd = np.mean([m["forward_pred_r2"] for m in seed_table_metrics])
    fwd_s = np.std([m["forward_pred_r2"] for m in seed_table_metrics])

    print(f"Static Single-Frame z_t : Position R^2 = {p_stat:.3f} ± {p_stat_s:.3f} | Velocity R^2 = {v_stat:.3f} ± {v_stat_s:.3f}")
    print(f"CPC Context c_t         : Position R^2 = {p_ctx:.3f} ± {p_ctx_s:.3f} | Velocity R^2 = {v_ctx:.3f} ± {v_ctx_s:.3f}")
    print(f"Next-State Forward Pred : Mean R^2     = {fwd:.3f} ± {fwd_s:.3f}")
    print("=" * 70)

    os.makedirs(args.results_dir, exist_ok=True)
    out_json = os.path.join(args.results_dir, "table_03_05_cpc_metrics.json")
    save_json({"seeds": args.seeds, "metrics": seed_table_metrics}, out_json)
    print(f"Saved Table 3.5 results -> {out_json}")


if __name__ == "__main__":
    main()
