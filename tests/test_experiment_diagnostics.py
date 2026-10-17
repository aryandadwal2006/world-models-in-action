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

    def test_latent_error_metrics_identical_vectors_are_perfect(self):
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        metrics = CPC._latent_error_metrics(vectors, vectors)
        self.assertAlmostEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["mean_cosine_similarity"], 1.0)
        self.assertEqual(metrics["num_predictions"], 2)

    def test_latent_error_metrics_detect_a_changed_prediction(self):
        predicted = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        metrics = CPC._latent_error_metrics(predicted, target)
        self.assertGreater(metrics["mse"], 0.0)
        self.assertLess(metrics["mean_cosine_similarity"], 1.0)


class DummyCPC(nn.Module):
    """Tiny GRU model for exercising teacher-forced and free-rollout metrics."""

    def __init__(self):
        super().__init__()
        self.k_steps = 3
        self.gru = nn.GRU(input_size=2, hidden_size=2, batch_first=True)
        self.predictor = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.predictor.weight.copy_(torch.eye(2))

    def encode_sequence(self, images):
        return nn.functional.normalize(images[:, :, 0, 0, :2], dim=-1)

    def predict_next_latent(self, context):
        return nn.functional.normalize(self.predictor(context), dim=-1)


class TestCPCLatentForecastDiagnostics(unittest.TestCase):
    def test_reports_teacher_forced_and_each_rollout_horizon(self):
        # Shape (batch=2, time=12, channels=1, height=1, width=2).
        base = torch.arange(1.0, 25.0).reshape(2, 12, 1, 1, 1)
        images = torch.cat([base, base.flip(dims=(1,))], dim=-1)
        loader = [{"images": images}]
        diagnostic = CPC.evaluate_latent_prediction_diagnostics(
            DummyCPC(),
            loader,
            torch.device("cpu"),
            rollout_start=5,
            max_horizon=3,
        )
        self.assertIn("teacher_forced_one_step", diagnostic)
        # seq_len=12, k_steps=3: train uses context indexes 4..8 for
        # one-step predictions, across both batch examples.
        self.assertEqual(
            diagnostic["teacher_forced_one_step"]["num_predictions"],
            10,
        )
        self.assertEqual(
            set(diagnostic["autoregressive_rollout_by_horizon"]),
            {"1", "2", "3"},
        )
        for item in diagnostic["autoregressive_rollout_by_horizon"].values():
            self.assertGreaterEqual(item["mse"], 0.0)
            self.assertLessEqual(item["mean_cosine_similarity"], 1.000001)

    def test_rejects_rollout_longer_than_sequence(self):
        base = torch.arange(1.0, 13.0).reshape(1, 12, 1, 1, 1)
        images = torch.cat([base, base.flip(dims=(1,))], dim=-1)
        with self.assertRaisesRegex(ValueError, "exceeds sequence length"):
            CPC.evaluate_latent_prediction_diagnostics(
                DummyCPC(),
                [{"images": images}],
                torch.device("cpu"),
                rollout_start=8,
                max_horizon=5,
            )


if __name__ == "__main__":
    unittest.main()
