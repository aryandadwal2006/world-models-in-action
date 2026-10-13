"""Temporal Contrastive Predictive Coding model used in Chapter 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
