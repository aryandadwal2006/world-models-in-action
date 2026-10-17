"""Regression tests for targeted Chapter 3 experiment diagnostics."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


def import_script(module_name, relative_path):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MECHANISMS = import_script("ch03_three_mechanisms_test", "ch03/07_three_mechanisms.py")
COLLAPSE = import_script("ch03_collapse_demo_test", "ch03/06_collapse_demo.py")
CPC = import_script("ch03_temporal_cpc_diagnostics_test", "ch03/11_temporal_cpc.py")


class TestMechanismHeadFaithfulness(unittest.TestCase):
    def test_method_specific_heads_include_batch_normalization(self):
        simsiam = MECHANISMS.SimSiamProjector(4, 8, 6)
        predictor = MECHANISMS.SimSiamPredictor(6, 8, 6)
        byol = MECHANISMS.BYOLHead(4, 8, 6)
        barlow = MECHANISMS.BarlowProjector(4, 8, 6)
        for module in (simsiam, predictor, byol, barlow):
            self.assertTrue(
                any(isinstance(layer, nn.BatchNorm1d) for layer in module.modules())
            )
        self.assertFalse(simsiam.net[-1].affine)
        self.assertIsNotNone(byol.net[0].bias)
        self.assertIsNone(byol.net[-1].bias)

    def test_method_specific_heads_preserve_batch_and_output_dimensions(self):
        x = torch.randn(8, 4)
        modules = (
            MECHANISMS.SimSiamProjector(4, 8, 6),
            MECHANISMS.BYOLHead(4, 8, 6),
            MECHANISMS.BarlowProjector(4, 8, 6),
        )
        for module in modules:
            result = module(x)
            self.assertEqual(tuple(result.shape), (8, 6))
            self.assertTrue(torch.isfinite(result).all().item())


class TestRepresentationDiagnostics(unittest.TestCase):
    def test_effective_rank_handles_constant_and_nonconstant_features(self):
        constant = np.ones((100, 4))
        self.assertEqual(
            MECHANISMS._feature_statistics(constant)["effective_rank"],
            0.0,
        )

        x = np.linspace(-1.0, 1.0, 100)
        features = np.column_stack([x, -x, np.sin(4 * x), np.cos(3 * x)])
        stats = MECHANISMS._feature_statistics(features)
        self.assertGreaterEqual(stats["effective_rank"], 1.0)
        self.assertLessEqual(stats["effective_rank"], 4.0)
        self.assertEqual(len(stats["per_dimension_std"]), 4)

    def test_collapse_diagnostics_report_zero_spread_for_constant_features(self):
        result = COLLAPSE._feature_diagnostics(np.ones((20, 5)))
        self.assertEqual(result["mean_dim_std"], 0.0)
        self.assertEqual(result["effective_rank"], 0.0)

class DummyCPC(nn.Module):
    """Tiny CPC-shaped model for teacher-forced and free-rollout metrics."""

    def __init__(self):
        super().__init__()
        self.k_steps = 3
        self.gru = nn.GRU(input_size=2, hidden_size=2, batch_first=True)
        self.w_k = nn.ParameterList(
            [nn.Parameter(torch.eye(2)) for _ in range(self.k_steps)]
        )

    def encode_sequence(self, images):
        return nn.functional.normalize(images[:, :, 0, 0, :2], dim=-1)

    def predict_next_latent(self, context):
        return nn.functional.normalize(context @ self.w_k[0].T, dim=-1)


class TestCPCFutureRetrieval(unittest.TestCase):
    def test_standard_infonce_retrieval_metrics_for_each_horizon(self):
        base = torch.arange(1.0, 49.0).reshape(4, 12, 1, 1, 1)
        images = torch.cat([base, base.flip(dims=(1,))], dim=-1)
        diagnostic = CPC.evaluate_cpc_future_retrieval(
            DummyCPC(),
            [{"images": images}],
            torch.device("cpu"),
        )
        self.assertEqual(
            set(diagnostic["by_horizon"]),
            {"1", "2", "3"},
        )
        self.assertIn("InfoNCE", diagnostic["method"])
        self.assertEqual(diagnostic["num_sequences_seen"], 4)
        for item in diagnostic["by_horizon"].values():
            self.assertTrue(np.isfinite(item["infonce_loss"]))
            self.assertGreaterEqual(item["infonce_loss"], 0.0)
            self.assertGreaterEqual(item["top1_accuracy"], 0.0)
            self.assertLessEqual(item["top1_accuracy"], 1.0)
            self.assertGreaterEqual(item["top5_accuracy"], item["top1_accuracy"])
            self.assertAlmostEqual(item["random_top1_chance"], 0.25)
            self.assertEqual(item["num_context_examples"], 4 * (12 - 3))

    def test_effective_rank_does_not_detect_representation_scale(self):
        values = np.column_stack([
            np.linspace(-1.0, 1.0, 100),
            np.linspace(1.0, -1.0, 100),
            np.sin(np.linspace(-1.0, 1.0, 100) * 4),
            np.cos(np.linspace(-1.0, 1.0, 100) * 3),
        ])
        ordinary = MECHANISMS._feature_statistics(values)
        tiny = MECHANISMS._feature_statistics(values * 1e-6)
        self.assertAlmostEqual(
            ordinary["effective_rank"],
            tiny["effective_rank"],
            places=6,
        )
        self.assertLess(tiny["mean_dim_std"], ordinary["mean_dim_std"] * 1e-5)


class TinyFrameEncoder(nn.Module):
    """Small deterministic-shape encoder for objective regression tests."""

    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.projection = nn.Linear(3, latent_dim)

    def forward(self, images):
        return self.projection(images.mean(dim=(-2, -1)))


class TestCanonicalCPCObjective(unittest.TestCase):
    def test_forward_uses_multihorizon_infonce_heads_without_auxiliary_mse_head(self):
        from worldmodels.models.temporal_cpc import TemporalCPC

        model = TemporalCPC(
            TinyFrameEncoder(latent_dim=4),
            latent_dim=4,
            context_dim=5,
            k_steps=2,
        )
        images = torch.randn(3, 8, 3, 8, 8)
        loss = model(images)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertFalse(hasattr(model, "predictor"))
        # Canonical CPC keeps the raw encoder scale rather than normalizing each z_t.
        encoded = model.encode_sequence(images)
        norms = torch.linalg.vector_norm(encoded, dim=-1)
        self.assertFalse(torch.allclose(norms, torch.ones_like(norms)))
        # Rollout helper is exactly the raw first bilinear prediction, without normalization.
        context = torch.randn(3, 5)
        expected_prediction = context @ model.w_k[0].T
        self.assertTrue(
            torch.allclose(model.predict_next_latent(context), expected_prediction)
        )
        for head in model.w_k:
            self.assertIsNotNone(head.grad)
            self.assertTrue(torch.isfinite(head.grad).all().item())


if __name__ == "__main__":
    unittest.main()
