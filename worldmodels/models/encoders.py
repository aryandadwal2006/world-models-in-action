"""Visual encoders and projection heads for world-model state representations.

Implements lightweight convolutional backbones and multi-layer perceptron (MLP)
projection heads designed for fast CPU execution and clean API composition.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from worldmodels.losses.reconstruction import masked_mse_loss


class ConvEncoder(nn.Module):
    """Convolutional neural network mapping 64x64 sensory frames to a compact latent vector.

    Designed with 3-4 strided convolutional stages to downsample from 64x64 to 4x4,
    followed by a dense linear bottleneck.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 16,
        base_channels: int = 32,
    ) -> None:
        """Initializes the ConvEncoder.

        Args:
            in_channels: Number of input image channels (e.g. 3 for RGB, 6 for stacked pairs).
            latent_dim: Dimensionality of the output representation vector z.
            base_channels: Feature channel multiplier.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        flat_dim = (base_channels * 8) * 4 * 4
        self.fc = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an input tensor (B, C, H, W) to latent vector z of shape (B, latent_dim)."""
        features = self.conv_net(x)
        flattened = torch.flatten(features, start_dim=1)
        z = self.fc(flattened)
        return z


class ProjectionHead(nn.Module):
    """Non-linear multi-layer perceptron (MLP) mapping latent z to projection space p.

    Used during self-supervised pre-training to prevent information loss in z
    caused by invariance constraints.
    """

    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 16,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder(nn.Module):
    """Unified encoder interface matching the Chapter 1 agent loop contract.

    Exposes `encode(obs)` and `update(state, obs)` methods, integrating optional
    temporal recurrent memory if specified.
    """

    def __init__(
        self,
        backbone: nn.Module,
        recurrent_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.recurrent_hidden_dim = recurrent_hidden_dim

        if recurrent_hidden_dim is not None:
            latent_dim = getattr(backbone, "latent_dim", 16)
            self.gru_cell = nn.GRUCell(latent_dim, recurrent_hidden_dim)
        else:
            self.gru_cell = None

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes observation into representation z."""
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        return self.backbone(obs)

    def update(
        self,
        state: Optional[torch.Tensor],
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Updates persistent state vector given a new observation.

        Args:
            state: Previous persistent state vector, or None if initial tick.
            obs: Incoming observation tensor.

        Returns:
            Updated persistent state vector.
        """
        z = self.encode(obs)
        if self.gru_cell is None:
            return z

        if state is None:
            batch_size = z.shape[0]
            device = z.device
            state = torch.zeros(batch_size, self.recurrent_hidden_dim, device=device)

        new_state = self.gru_cell(z, state)
        return new_state


class SimpleMAE(nn.Module):
    """Lightweight patch-based Masked Autoencoder (He et al., 2022).

    Un-shuffles visible tokens and learnable mask tokens using ids_restore in the decoder,
    while exposing a pooled latent bottleneck z for linear state probing.
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.mask_ratio = mask_ratio

        self.num_patches = (img_size // patch_size) ** 2  # (64/8)^2 = 64
        self.patch_dim = in_channels * patch_size * patch_size  # 3*8*8 = 192

        # Patch projection and 1D learnable position embeddings
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

        # Encoder backbone (2 Transformer encoder layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Representation bottleneck: global average pool over visible tokens -> latent_dim
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        # Decoder
        self.decoder_proj = nn.Linear(latent_dim, embed_dim)
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
        """Converts (B, C, H, W) to (B, num_patches, patch_dim)."""
        p = self.patch_size
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = torch.einsum("bchpwq->bhwcpq", x)
        patches = x.reshape(b, self.num_patches, self.patch_dim)
        return patches

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """Converts (B, num_patches, patch_dim) back to (B, C, H, W)."""
        p = self.patch_size
        h = w = self.img_size // p
        b = patches.shape[0]
        x = patches.reshape(b, h, w, self.in_channels, p, p)
        x = torch.einsum("bhwcpq->bchpwq", x)
        imgs = x.reshape(b, self.in_channels, self.img_size, self.img_size)
        return imgs

    def forward_encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes visible patches, returning (z_latent, visible_tokens, mask, ids_restore)."""
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.pos_embed

        b, n, d = tokens.shape
        len_keep = int(n * (1.0 - self.mask_ratio))

        # Random masking per sample
        noise = torch.rand(b, n, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        visible_tokens = torch.gather(tokens, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))

        # Generate binary mask: 0 = visible, 1 = masked
        mask = torch.ones(b, n, device=x.device)
        mask[:, :len_keep] = 0.0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Transformer encoder over visible tokens
        encoded_visible = self.encoder_transformer(visible_tokens)
        z = self.to_latent(encoded_visible.mean(dim=1))
        return z, encoded_visible, mask, ids_restore

    def forward_decoder(
        self,
        encoded_visible: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstructs all patch tokens by inserting learnable mask tokens and un-shuffling."""
        b = encoded_visible.shape[0]
        len_keep = encoded_visible.shape[1]
        n = self.num_patches

        # Append mask tokens to encoded visible tokens
        mask_tokens = self.mask_token.repeat(b, n - len_keep, 1)
        x_ = torch.cat([encoded_visible, mask_tokens], dim=1)

        # Un-shuffle back to original 2D grid order using ids_restore
        tokens = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, self.embed_dim))
        tokens = tokens + self.pos_embed
        decoded = self.decoder_transformer(tokens)
        pred_patches = self.pred_head(decoded)
        return pred_patches

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full MAE forward pass returning (loss, pred_patches, mask)."""
        patches = self.patchify(x)
        z, encoded_visible, mask, ids_restore = self.forward_encoder(x)
        pred_patches = self.forward_decoder(encoded_visible, ids_restore)
        loss = masked_mse_loss(patches, pred_patches, mask)
        return loss, pred_patches, mask

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference encoder extracting latent code z without random masking."""
        patches = self.patchify(x)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder_transformer(tokens)
        z = self.to_latent(encoded.mean(dim=1))
        return z


class MAEEncoder(nn.Module):
    """Adapter wrapping SimpleMAE to present a standard encode(x) callable interface."""

    def __init__(self, mae: SimpleMAE) -> None:
        super().__init__()
        self.mae = mae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mae.encode(x)

