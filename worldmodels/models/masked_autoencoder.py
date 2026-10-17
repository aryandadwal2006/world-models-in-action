"""Patch-based masked autoencoder used in Chapter 3."""

from __future__ import annotations

import torch
import torch.nn as nn

from worldmodels.losses.reconstruction import masked_mse_loss


class SimpleMAE(nn.Module):
    """Small patch-based masked autoencoder for the DMC images.

    The mean-pooled final encoder tokens are the representation exposed to
    the linear probe. The decoder deliberately operates on per-patch encoder
    tokens, not on the pooled vector, matching the basic MAE design in which
    downstream features and reconstruction use distinct paths.

    ``latent_dim`` remains accepted for call-site compatibility, but does not
    control MAE probe features: their dimension is ``embed_dim``.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
        latent_dim: int = 16,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        # Backwards-compatible field: probe features use embed_dim directly.
        self.latent_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        if int(self.num_patches * (1.0 - mask_ratio)) < 1:
            raise ValueError("mask_ratio must leave at least one visible patch")
        self.patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.encoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=1)
        self.pred_head = nn.Linear(embed_dim, self.patch_dim)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        b, c, h, w = x.shape
        if c != self.in_channels or h != self.img_size or w != self.img_size:
            raise ValueError("Input shape does not match MAE configuration")
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = torch.einsum("bchpwq->bhwcpq", x)
        return x.reshape(b, self.num_patches, self.patch_dim)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = self.img_size // p
        b = patches.shape[0]
        x = patches.reshape(b, h, w, self.in_channels, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        return x.reshape(b, self.in_channels, self.img_size, self.img_size)

    def forward_encoder(self, x: torch.Tensor):
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.encoder_pos_embed
        b, n, d = tokens.shape
        len_keep = int(n * (1.0 - self.mask_ratio))

        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        visible = torch.gather(tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))

        mask = torch.ones(b, n, device=x.device)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, 1, ids_restore)

        encoded_visible = self.encoder_transformer(visible)
        # Pool trained encoder tokens directly; reconstruction does not train
        # a separate projection head.
        z = encoded_visible.mean(dim=1)
        return z, encoded_visible, mask, ids_restore

    def forward_decoder(self, encoded_visible: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        b, len_keep, _ = encoded_visible.shape
        n = self.num_patches
        mask_tokens = self.mask_token.expand(b, n - len_keep, -1)
        tokens_shuffled = torch.cat([encoded_visible, mask_tokens], dim=1)
        tokens = torch.gather(
            tokens_shuffled,
            1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )
        tokens = tokens + self.decoder_pos_embed
        decoded = self.decoder_transformer(tokens)
        return self.pred_head(decoded)

    def forward(self, x: torch.Tensor):
        patches = self.patchify(x)
        _, encoded_visible, mask, ids_restore = self.forward_encoder(x)
        pred_patches = self.forward_decoder(encoded_visible, ids_restore)
        loss = masked_mse_loss(patches, pred_patches, mask)
        return loss, pred_patches, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.encoder_pos_embed
        encoded = self.encoder_transformer(tokens)
        return encoded.mean(dim=1)


class MAEEncoder(nn.Module):
    """Adapter exposing the MAE downstream representation as ``forward``."""

    def __init__(self, mae: SimpleMAE) -> None:
        super().__init__()
        self.mae = mae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mae.encode(x)
