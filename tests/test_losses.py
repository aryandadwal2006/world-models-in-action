"""Unit tests for contrastive and reconstruction loss functions."""

import unittest
import torch
import torch.nn as nn

from worldmodels.losses.contrastive import (
    nt_xent_loss,
    simsiam_loss,
    update_target_ema,
    barlow_twins_loss,
    vicreg_loss,
)
from worldmodels.losses.reconstruction import masked_mse_loss


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 8
        self.dim = 16
        self.z1 = torch.randn(self.batch_size, self.dim)
        self.z2 = torch.randn(self.batch_size, self.dim)

    def test_nt_xent_loss_basic(self):
        loss = nt_xent_loss(self.z1, self.z2, temperature=0.5)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_nt_xent_identical_views_lower_loss(self):
        loss_random = nt_xent_loss(self.z1, self.z2, temperature=0.5).item()
        # Identical views should produce much lower loss than independent random views
        loss_identical = nt_xent_loss(self.z1, self.z1, temperature=0.5).item()
        self.assertLess(loss_identical, loss_random)

    def test_simsiam_loss(self):
        p1 = torch.randn(self.batch_size, self.dim, requires_grad=True)
        p2 = torch.randn(self.batch_size, self.dim, requires_grad=True)
        z1 = torch.randn(self.batch_size, self.dim)
        z2 = torch.randn(self.batch_size, self.dim)

        loss = simsiam_loss(p1, z2, p2, z1)
        self.assertTrue(torch.is_tensor(loss))
        # Negative cosine similarity lies in [-1, 1]
        self.assertGreaterEqual(loss.item(), -1.01)
        self.assertLessEqual(loss.item(), 1.01)

    def test_update_target_ema(self):
        online = nn.Linear(10, 10)
        target = nn.Linear(10, 10)

        with torch.no_grad():
            for p in online.parameters():
                p.fill_(2.0)
            for p in target.parameters():
                p.fill_(1.0)

        update_target_ema(online, target, decay=0.9)
        # target = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        self.assertAlmostEqual(target.weight.data[0, 0].item(), 1.1, places=5)

    def test_barlow_twins_identical_views_have_zero_loss(self):
        loss = barlow_twins_loss(self.z1, self.z1, lambd=0.005)
        self.assertLess(loss.item(), 1e-8)

    def test_barlow_twins_requires_at_least_two_examples(self):
        with self.assertRaises(ValueError):
            barlow_twins_loss(self.z1[:1], self.z2[:1])

    def test_barlow_twins_loss(self):
        loss = barlow_twins_loss(self.z1, self.z2, lambd=0.005)
        self.assertTrue(torch.is_tensor(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_vicreg_loss_components(self):
        total, sim, var, cov = vicreg_loss(self.z1, self.z2)
        self.assertTrue(torch.is_tensor(total))
        self.assertTrue(torch.is_tensor(sim))
        self.assertTrue(torch.is_tensor(var))
        self.assertTrue(torch.is_tensor(cov))
        self.assertGreater(total.item(), 0.0)
        self.assertGreater(sim.item(), 0.0)

    def test_masked_mse_loss(self):
        target = torch.ones(2, 4, 8)
        pred = torch.zeros(2, 4, 8)
        mask = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

        loss = masked_mse_loss(target, pred, mask)
        # All masked patches have error (1 - 0)^2 = 1.0
        self.assertAlmostEqual(loss.item(), 1.0, places=5)

    def test_masked_mse_loss_empty_mask(self):
        target = torch.ones(2, 4, 8)
        pred = torch.zeros(2, 4, 8, requires_grad=True)
        mask = torch.zeros(2, 4)

        loss = masked_mse_loss(target, pred, mask)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(pred.grad)
        self.assertEqual(pred.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
