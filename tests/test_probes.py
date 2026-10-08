"""Tests for stable and angle-aware Chapter 3 probes."""

import unittest

import numpy as np

from worldmodels.eval.probes import (
    LinearProbe,
    StateLinearProbe,
    circular_r2_score,
    compute_r2_score,
    compute_structured_r2_score,
    expanded_state_dim,
    infer_angular_position_indices,
    infer_angular_position_periods,
    transform_state_targets_np,
)


class TestProbes(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)

        self.x = rng.standard_normal(
            (200, 8)
        )

        self.w_true = rng.standard_normal(
            (8, 2)
        )

        self.y = (
            self.x @ self.w_true
            + 0.05
            * rng.standard_normal(
                (200, 2)
            )
        )

    def test_compute_r2_score_perfect(self):
        y = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )

        np.testing.assert_allclose(
            compute_r2_score(y, y),
            np.ones(2),
        )

    def test_compute_r2_score_mean_prediction(self):
        y = np.array(
            [
                [1.0],
                [3.0],
                [5.0],
            ]
        )

        pred = np.array(
            [
                [3.0],
                [3.0],
                [3.0],
            ]
        )

        self.assertAlmostEqual(
            float(
                compute_r2_score(
                    y,
                    pred,
                )[0]
            ),
            0.0,
            places=7,
        )

    def test_linear_probe_fit_and_predict(self):
        probe = LinearProbe(
            alpha=1e-4
        ).fit(
            self.x[:150],
            self.y[:150],
        )

        r2, mean_r2 = probe.score(
            self.x[150:],
            self.y[150:],
        )

        self.assertGreater(
            mean_r2,
            0.95,
        )

        self.assertTrue(
            np.all(r2 > 0.95)
        )

    def test_feature_standardization_handles_scale_difference(self):
        rng = np.random.default_rng(7)

        x = rng.normal(
            size=(200, 3)
        )

        x[:, 0] *= 1e6
        x[:, 1] *= 1e-6

        y = np.column_stack(
            [
                2.0
                * x[:, 0]
                / 1e6
                + 0.5
                * x[:, 1]
                / 1e-6,
                -x[:, 2],
            ]
        )

        probe = LinearProbe(
            alpha=1e-3
        ).fit(
            x[:150],
            y[:150],
        )

        r2, mean_r2 = probe.score(
            x[150:],
            y[150:],
        )

        self.assertTrue(
            np.isfinite(
                probe.weights
            ).all()
        )

        self.assertGreater(
            mean_r2,
            0.9,
        )

        self.assertTrue(
            np.all(r2 > 0.9)
        )

    def test_high_dimensional_fit_is_finite(self):
        rng = np.random.default_rng(123)

        x = rng.standard_normal(
            (40, 256)
        )

        y = rng.standard_normal(
            (40, 6)
        )

        probe = LinearProbe(
            alpha=1e-3
        ).fit(
            x[:30],
            y[:30],
        )

        pred = probe.predict(
            x[30:]
        )

        self.assertTrue(
            np.isfinite(
                probe.weights
            ).all()
        )

        self.assertTrue(
            np.isfinite(
                pred
            ).all()
        )

    def test_canonical_angular_indices(self):
        self.assertEqual(
            infer_angular_position_indices(
                4
            ),
            (1,),
        )

        self.assertEqual(
            infer_angular_position_indices(
                6
            ),
            (2,),
        )

        self.assertEqual(
            infer_angular_position_indices(
                18
            ),
            tuple(range(2, 9)),
        )

    def test_finger_spinner_uses_pi_period(self):
        periods = infer_angular_position_periods(6)
        self.assertAlmostEqual(periods[2], np.pi)
        self.assertAlmostEqual(
            infer_angular_position_periods(4)[1],
            2.0 * np.pi,
        )

    def test_circular_score_respects_pi_period(self):
        true = np.array([-1.4, -0.5, 0.2, 1.45])
        pred = true + np.pi
        self.assertAlmostEqual(
            circular_r2_score(true, pred, period=np.pi),
            1.0,
            places=10,
        )

    def test_pi_periodic_probe_decodes_rendered_orientation(self):
        rng = np.random.default_rng(2026)
        theta = rng.uniform(-np.pi, np.pi, size=500)
        # The rendered spinner is symmetric under theta -> theta + pi.
        x = np.column_stack([
            np.sin(2.0 * theta),
            np.cos(2.0 * theta),
            rng.normal(size=len(theta)),
        ])
        y = theta[:, None]
        probe = StateLinearProbe(
            alpha=1e-8,
            angular_position_indices=[0],
            angular_position_periods={0: np.pi},
        ).fit(x[:350], y[:350])
        r2, _ = probe.score(x[350:], y[350:])
        self.assertGreater(r2[0], 0.99)

    def test_sin_cos_expansion(self):
        states = np.array(
            [
                [0.0, np.pi / 2],
                [1.0, -np.pi / 2],
            ]
        )

        expanded = transform_state_targets_np(
            states,
            [1],
        )

        self.assertEqual(
            expanded.shape,
            (2, 3),
        )

        np.testing.assert_allclose(
            expanded,
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, -1.0, 0.0],
                ]
            ),
            atol=1e-7,
        )

        self.assertEqual(
            expanded_state_dim(
                2,
                [1],
            ),
            3,
        )

    def test_circular_score_is_wrap_invariant(self):
        true = np.array(
            [
                np.pi - 0.3,
                -np.pi + 0.2,
                0.4,
                -0.7,
            ]
        )

        pred = true + 2.0 * np.pi

        self.assertAlmostEqual(
            circular_r2_score(
                true,
                pred,
            ),
            1.0,
            places=10,
        )

        np.testing.assert_allclose(
            compute_structured_r2_score(
                true[:, None],
                pred[:, None],
                [0],
            ),
            np.ones(1),
        )

    def test_state_linear_probe_decodes_angle(self):
        rng = np.random.default_rng(11)

        theta = rng.uniform(
            -np.pi,
            np.pi,
            size=400,
        )

        x = np.column_stack(
            [
                np.sin(theta),
                np.cos(theta),
                rng.normal(
                    size=len(theta)
                ),
            ]
        )

        y = theta[:, None]

        probe = StateLinearProbe(
            alpha=1e-8,
            angular_position_indices=[0],
        )

        probe.fit(
            x[:300],
            y[:300],
        )

        r2, mean_r2 = probe.score(
            x[300:],
            y[300:],
        )

        self.assertGreater(
            r2[0],
            0.99,
        )

        self.assertGreater(
            mean_r2,
            0.99,
        )

    def test_unfitted_probe_raises(self):
        with self.assertRaises(
            RuntimeError
        ):
            LinearProbe().predict(
                self.x
            )

    def test_unfitted_state_probe_raises(self):
        with self.assertRaises(
            RuntimeError
        ):
            StateLinearProbe(
                angular_position_indices=[0]
            ).predict(
                self.x
            )


if __name__ == "__main__":
    unittest.main()