"""Unit tests for Table 3.4's result migration and aggregation contract."""

import copy
import unittest

import torch
import torch.nn as nn

from ch03.table_03_04_comparison import (
    evaluate_trained_state_head,
    EXCLUDED_STATE_VARIABLES,
    METHOD_DISPLAY_NAMES,
    METHOD_NAMES,
    REQUIRED_TASKS,
    _migrate_version3_progress,
    _migrate_version5_mae_representation_progress,
    _migrate_version6_supervised_head_progress,
    _resummarize_saved_result,
)


class TestTable0304Metrics(unittest.TestCase):
    def test_trained_supervised_head_is_scored_in_physical_state_space(self):
        class StateFeatureEncoder(nn.Module):
            def forward(self, states):
                angle = states[:, 1]
                return torch.stack(
                    [
                        states[:, 0],
                        torch.sin(angle),
                        torch.cos(angle),
                        states[:, 2],
                        states[:, 3],
                    ],
                    dim=1,
                )

        states = torch.tensor(
            [
                [-0.8, -2.2, -0.7, 0.5],
                [-0.3, -0.9, 0.2, -0.1],
                [0.1, 0.2, 0.6, -0.8],
                [0.6, 1.1, -0.4, 0.3],
                [1.0, 2.4, 0.9, 0.8],
            ],
            dtype=torch.float32,
        )
        result = evaluate_trained_state_head(
            StateFeatureEncoder(),
            nn.Identity(),
            [{"image": states, "physics_state": states}],
            torch.zeros(5),
            torch.ones(5),
            state_dim=4,
            device=torch.device("cpu"),
        )

        self.assertAlmostEqual(result["position_mean_r2"], 1.0, places=5)
        self.assertAlmostEqual(result["velocity_mean_r2"], 1.0, places=5)
        self.assertAlmostEqual(result["mean_r2"], 1.0, places=5)

    def test_supervised_row_label_matches_the_reported_metric(self):
        self.assertIn("Supervised State-Prediction Reference", METHOD_NAMES)
        self.assertEqual(
            METHOD_DISPLAY_NAMES["Supervised State-Prediction Reference"],
            "Supervised-pretrained encoder + frozen-feature linear probe",
        )

    def test_version5_migration_clears_only_mae_results(self):
        old_signature = {
            "experiment_version": 5,
            "latent_dim": 16,
            "tasks": ["cartpole_balance", "finger_spin"],
        }
        new_signature = copy.deepcopy(old_signature)
        new_signature["experiment_version"] = 6
        new_signature["mae_representation"] = {
            "source": "mean_pooled_final_encoder_tokens",
            "feature_dim": 64,
            "projection": None,
        }
        results = {
            task: {
                method: {"0": {"mean_r2": 0.25}}
                for method in METHOD_NAMES
            }
            for task, _state_dim in REQUIRED_TASKS
        }
        progress = {"experiment": old_signature, "results": results}

        migrated = _migrate_version5_mae_representation_progress(
            progress, new_signature
        )

        self.assertIsNotNone(migrated)
        self.assertEqual(migrated["experiment"], new_signature)
        for task, _state_dim in REQUIRED_TASKS:
            self.assertEqual(
                migrated["results"][task]["Masked Autoencoder (MAE)"], {}
            )
            self.assertEqual(
                migrated["results"][task]["Contrastive (NT-Xent)"],
                {"0": {"mean_r2": 0.25}},
            )
            self.assertEqual(
                migrated["results"][task]["VICReg"],
                {"0": {"mean_r2": 0.25}},
            )
            self.assertIn(
                task,
                migrated["legacy_results_before_mae_representation_fix"][
                    "version5_mae_results"
                ],
            )

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

    def test_version5_to_version7_migration_clears_supervised_and_mae_only(self):
        old_signature = {
            "experiment_version": 5,
            "epochs": 12,
            "batch_size": 64,
            "lr": 1e-3,
            "latent_dim": 16,
            "max_train_samples": 3500,
            "max_val_samples": 1000,
            "seeds": [0, 1, 2],
            "tasks": ["cartpole_balance", "finger_spin"],
            "augmentation": {
                "max_shift": 3,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
            },
            "sample_selection": "episode_stratified_even_within_episode",
            "probe": {"feature_standardization": True},
            "datasets": {"same": "metadata"},
        }
        new_signature = copy.deepcopy(old_signature)
        new_signature["experiment_version"] = 7
        new_signature["mae_representation"] = {
            "source": "mean_pooled_final_encoder_tokens",
            "feature_dim": 64,
            "projection": None,
        }
        results = {
            task: {
                method: {"0": {"mean_r2": 0.25}}
                for method in METHOD_NAMES
            }
            for task, _state_dim in REQUIRED_TASKS
        }
        progress = {"experiment": old_signature, "results": results}

        migrated = _migrate_version5_mae_representation_progress(
            progress, new_signature
        )

        self.assertIsNotNone(migrated)
        for task, _state_dim in REQUIRED_TASKS:
            self.assertEqual(migrated["results"][task][METHOD_NAMES[0]], {})
            self.assertEqual(migrated["results"][task]["Masked Autoencoder (MAE)"], {})
            for method in METHOD_NAMES[1:3]:
                self.assertEqual(
                    migrated["results"][task][method],
                    {"0": {"mean_r2": 0.25}},
                )
            self.assertIn(
                task,
                migrated["legacy_results_before_trained_head_diagnostic"][
                    "version5_supervised_probe_only_results"
                ],
            )

    def test_version6_migration_retrains_only_supervised_head_arm(self):
        old_signature = {
            "experiment_version": 6,
            "epochs": 12,
            "batch_size": 64,
            "lr": 1e-3,
            "latent_dim": 16,
            "mae_representation": {
                "source": "mean_pooled_final_encoder_tokens",
                "feature_dim": 64,
                "projection": None,
            },
            "max_train_samples": 3500,
            "max_val_samples": 1000,
            "seeds": [0, 1, 2],
            "tasks": ["cartpole_balance", "finger_spin"],
            "augmentation": {
                "max_shift": 3,
                "brightness_range": 0.1,
                "contrast_range": 0.1,
            },
            "sample_selection": "episode_stratified_even_within_episode",
            "probe": {"feature_standardization": True},
            "datasets": {"same": "metadata"},
        }
        new_signature = copy.deepcopy(old_signature)
        new_signature["experiment_version"] = 7
        results = {
            task: {
                method: {"0": {"mean_r2": 0.25}}
                for method in METHOD_NAMES
            }
            for task, _state_dim in REQUIRED_TASKS
        }
        progress = {"experiment": old_signature, "results": results}

        migrated = _migrate_version6_supervised_head_progress(
            progress, new_signature
        )

        self.assertIsNotNone(migrated)
        self.assertEqual(migrated["experiment"], new_signature)
        for task, _state_dim in REQUIRED_TASKS:
            self.assertEqual(
                migrated["results"][task][METHOD_NAMES[0]],
                {},
            )
            for method in METHOD_NAMES[1:]:
                self.assertEqual(
                    migrated["results"][task][method],
                    {"0": {"mean_r2": 0.25}},
                )
            self.assertEqual(
                migrated["legacy_results_before_trained_head_diagnostic"][
                    "version6_supervised_probe_only_results"
                ][task][METHOD_NAMES[0]],
                {"0": {"mean_r2": 0.25}},
            )

    def test_version3_migration_archives_all_biased_sampling_results(self):
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
        for task, _state_dim in REQUIRED_TASKS:
            for method in METHOD_NAMES:
                self.assertEqual(
                    migrated["results"][task][method],
                    {},
                )

        legacy = migrated["legacy_results_before_sampling_fix"]["version3_results"]
        self.assertEqual(
            legacy["finger_spin"][METHOD_NAMES[1]]["0"]["r2_per_variable"][2],
            -1000.0,
        )
        self.assertEqual(
            len(legacy["cartpole_balance"][METHOD_NAMES[0]]["0"]["r2_per_variable"]),
            4,
        )


if __name__ == "__main__":
    unittest.main()
