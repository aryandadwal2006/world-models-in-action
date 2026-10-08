"""Unit tests for encoder architectures (worldmodels.models.encoders)."""

import unittest
import torch

from worldmodels.models.encoders import (
    ConvEncoder,
    ProjectionHead,
    Encoder,
    SimpleMAE,
    MAEEncoder,
)


class TestEncoders(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.latent_dim = 16
        self.x = torch.rand(self.batch_size, 3, 64, 64)

    def test_conv_encoder_forward(self):
        encoder = ConvEncoder(in_channels=3, latent_dim=self.latent_dim)
        z = encoder(self.x)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

    def test_conv_encoder_frame_stack(self):
        # 2 frames stacked: 6 channels
        x_stacked = torch.rand(self.batch_size, 6, 64, 64)
        encoder = ConvEncoder(in_channels=6, latent_dim=self.latent_dim)
        z = encoder(x_stacked)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

    def test_projection_head(self):
        head = ProjectionHead(in_dim=self.latent_dim, hidden_dim=32, out_dim=self.latent_dim)
        z = torch.randn(self.batch_size, self.latent_dim)
        p = head(z)
        self.assertEqual(p.shape, (self.batch_size, self.latent_dim))

    def test_encoder_agent_interface_stateless(self):
        backbone = ConvEncoder(in_channels=3, latent_dim=self.latent_dim)
        enc = Encoder(backbone=backbone, recurrent_hidden_dim=None)

        obs = torch.rand(3, 64, 64)  # single observation (C, H, W)
        z = enc.encode(obs)
        self.assertEqual(z.shape, (1, self.latent_dim))

        state = enc.update(None, obs)
        self.assertEqual(state.shape, (1, self.latent_dim))

    def test_encoder_agent_interface_recurrent(self):
        backbone = ConvEncoder(in_channels=3, latent_dim=self.latent_dim)
        recurrent_dim = 32
        enc = Encoder(backbone=backbone, recurrent_hidden_dim=recurrent_dim)

        obs = torch.rand(1, 3, 64, 64)
        state = None
        for _ in range(3):
            state = enc.update(state, obs)
            self.assertEqual(state.shape, (1, recurrent_dim))

    def test_simple_mae_forward_and_reconstruction(self):
        mae = SimpleMAE(img_size=64, patch_size=8, latent_dim=self.latent_dim, mask_ratio=0.75)
        loss, pred_patches, mask = mae(self.x)

        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(pred_patches.shape, (self.batch_size, 64, 3 * 8 * 8))
        self.assertEqual(mask.shape, (self.batch_size, 64))

        z = mae.encode(self.x)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        wrapper = MAEEncoder(mae)
        z_wrapped = wrapper(self.x)
        self.assertEqual(z_wrapped.shape, (self.batch_size, self.latent_dim))


if __name__ == "__main__":
    unittest.main()
