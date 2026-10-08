"""Unit tests for linear probes and evaluation metrics (worldmodels.eval.probes)."""

import unittest
import numpy as np

from worldmodels.eval.probes import LinearProbe, compute_r2_score


class TestProbes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n_samples = 200
        self.n_features = 8
        self.n_targets = 2

        self.x = np.random.randn(self.n_samples, self.n_features)
        # Construct linear target with known weights plus small noise
        self.w_true = np.random.randn(self.n_features, self.n_targets)
        self.y = self.x @ self.w_true + 0.05 * np.random.randn(self.n_samples, self.n_targets)

    def test_compute_r2_score_perfect(self):
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        r2 = compute_r2_score(y_true, y_true)
        np.testing.assert_allclose(r2, np.array([1.0, 1.0]), atol=1e-5)

    def test_compute_r2_score_mean_prediction(self):
        y_true = np.array([[1.0], [3.0], [5.0]])
        y_pred = np.array([[3.0], [3.0], [3.0]])  # constant mean
        r2 = compute_r2_score(y_true, y_pred)
        self.assertAlmostEqual(r2[0], 0.0, places=5)

    def test_linear_probe_fit_and_predict(self):
        probe = LinearProbe(alpha=1e-4)
        probe.fit(self.x[:150], self.y[:150])

        r2_per_var, mean_r2 = probe.score(self.x[150:], self.y[150:])
        # Should achieve very high R^2 (> 0.95) on linear data
        self.assertGreater(mean_r2, 0.95)
        for r2 in r2_per_var:
            self.assertGreater(r2, 0.95)

    def test_linear_probe_unfitted_raises(self):
        probe = LinearProbe()
        with self.assertRaises(RuntimeError):
            probe.predict(self.x)


if __name__ == "__main__":
    unittest.main()
