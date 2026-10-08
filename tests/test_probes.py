"""Tests for stable Ridge probes and evaluation R^2."""

import unittest

import numpy as np

from worldmodels.eval.probes import LinearProbe, compute_r2_score


class TestProbes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.standard_normal((200, 8))
        self.w_true = rng.standard_normal((8, 2))
        self.y = self.x @ self.w_true + 0.05 * rng.standard_normal((200, 2))

    def test_compute_r2_score_perfect(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_allclose(compute_r2_score(y, y), np.ones(2))

    def test_compute_r2_score_mean_prediction(self):
        y = np.array([[1.0], [3.0], [5.0]])
        pred = np.array([[3.0], [3.0], [3.0]])
        self.assertAlmostEqual(float(compute_r2_score(y, pred)[0]), 0.0, places=7)

    def test_linear_probe_fit_and_predict(self):
        probe = LinearProbe(alpha=1e-4).fit(self.x[:150], self.y[:150])
        r2, mean_r2 = probe.score(self.x[150:], self.y[150:])
        self.assertGreater(mean_r2, 0.95)
        self.assertTrue(np.all(r2 > 0.95))

    def test_high_dimensional_fit_is_finite(self):
        rng = np.random.default_rng(123)
        x = rng.standard_normal((40, 256))
        y = rng.standard_normal((40, 6))
        probe = LinearProbe(alpha=1e-3).fit(x[:30], y[:30])
        pred = probe.predict(x[30:])
        self.assertTrue(np.isfinite(probe.weights).all())
        self.assertTrue(np.isfinite(pred).all())

    def test_unfitted_probe_raises(self):
        with self.assertRaises(RuntimeError):
            LinearProbe().predict(self.x)


if __name__ == "__main__":
    unittest.main()
