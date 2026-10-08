"""Unit tests for Table 3.4's result migration and aggregation contract."""

import copy
import unittest

from ch03.table_03_04_comparison import (
    EXCLUDED_STATE_VARIABLES,
    METHOD_NAMES,
    REQUIRED_TASKS,
    _migrate_version3_progress,
    _resummarize_saved_result,
)


class TestTable0304Metrics(unittest.TestCase):
    def test_finger_spinner_angle_is_excluded_from_aggregate(self):
        result = {
            "r2_per_variable": [
                0.6,
                -0.5,
                -1000.0,
                -0.1,
                -0.2,
                -0.3,
            ]
        }

        _resummarize_saved_result(
            "finger_spin",
            result,
            legacy_full_turn_score=-1000.0,
        )

        self.assertIsNone(result["r2_per_variable"][2])
        self.assertEqual(
            result["legacy_excluded_spinner_angle_r2_full_turn"],
            -1000.0,
        )
        self.assertAlmostEqual(result["position_mean_r2"], 0.05)
        self.assertAlmostEqual(result["velocity_mean_r2"], -0.2)
        self.assertAlmostEqual(result["mean_r2"], -0.1)
        self.assertEqual(
            result["excluded_state_variables"],
            [EXCLUDED_STATE_VARIABLES["finger_spin"]],
        )

    def test_version3_migration_requeues_only_finger_supervised_runs(self):
        old_signature = {
            "experiment_version": 3,
            "probe": {
                "feature_standardization": True,
                "angular_position_encoding": "sin_cos",
                "angular_r2": "circular_chordal",
            },
            "shared_config": "same",
        }
        new_signature = copy.deepcopy(old_signature)
        new_signature["experiment_version"] = 4
        new_signature["probe"] = {
            "angular_position_encoding": "sin_cos_with_rendered_period"
        }

        results = {}
        for task, state_dim in REQUIRED_TASKS:
            results[task] = {}
            for method in METHOD_NAMES:
                scores = [0.1] * state_dim
                if task == "finger_spin":
                    scores[2] = -1000.0
                results[task][method] = {
                    "0": {
                        "r2_per_variable": scores,
                        "mean_r2": -1.0,
                        "position_mean_r2": -1.0,
                        "velocity_mean_r2": -1.0,
                    }
                }

        progress = {
            "experiment": old_signature,
            "results": results,
        }

        migrated = _migrate_version3_progress(
            progress,
            new_signature,
        )

        self.assertIsNotNone(migrated)
        self.assertEqual(migrated["experiment"], new_signature)
        self.assertEqual(
            migrated["results"]["finger_spin"][
                METHOD_NAMES[0]
            ],
            {},
        )
        self.assertIn(
            "finger_spin_supervised_reference",
            migrated["legacy_results_before_period_fix"],
        )
        retained = migrated["results"]["finger_spin"][METHOD_NAMES[1]]["0"]
        self.assertIsNone(retained["r2_per_variable"][2])
        self.assertEqual(
            retained["legacy_excluded_spinner_angle_r2_full_turn"],
            -1000.0,
        )


if __name__ == "__main__":
    unittest.main()
