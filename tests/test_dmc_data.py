"""Tests for DMC dataset stacking and episode-boundary handling."""

import unittest

import numpy as np
import torch

from worldmodels.data.dmc_data import DMCDataset, select_episode_stratified_indices


class TestDMCData(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.frames = rng.integers(0, 255, (20, 64, 64, 3), dtype=np.uint8)
        self.physics = rng.standard_normal((20, 4), dtype=np.float32)
        self.episode_ids = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    def test_single_frame_dataset(self):
        ds = DMCDataset(self.frames, self.physics, self.episode_ids)
        self.assertEqual(len(ds), 20)
        item = ds[0]
        self.assertEqual(item["image"].shape, (3, 64, 64))
        self.assertEqual(item["physics_state"].shape, (4,))
        self.assertTrue(torch.is_tensor(item["image"]))

    def test_episode_stratified_indices_cover_both_episodes(self):
        selected = select_episode_stratified_indices(
            self.episode_ids,
            max_samples=8,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(set(self.episode_ids[selected]), {0, 1})
        self.assertEqual(
            int(np.sum(self.episode_ids[selected] == 0)),
            4,
        )
        self.assertEqual(
            int(np.sum(self.episode_ids[selected] == 1)),
            4,
        )

    def test_short_episodes_do_not_waste_sample_budget(self):
        episode_ids = np.array([0] + [1] * 20 + [2] * 20)
        selected = select_episode_stratified_indices(
            episode_ids,
            max_samples=20,
        )
        self.assertEqual(len(selected), 20)
        counts = {
            episode: int(np.sum(episode_ids[selected] == episode))
            for episode in np.unique(episode_ids)
        }
        self.assertEqual(counts[0], 1)
        self.assertEqual(counts[1], 10)
        self.assertEqual(counts[2], 9)

    def test_sample_indices_preserve_original_temporal_stacks(self):
        selected = np.array([8, 11, 18], dtype=np.int64)
        ds = DMCDataset(
            self.frames,
            self.physics,
            self.episode_ids,
            frame_stack=2,
            strict_frame_stack=True,
            sample_indices=selected,
        )
        self.assertEqual(len(ds), 3)
        np.testing.assert_array_equal(
            [int(ds[i]["index"].item()) for i in range(len(ds))],
            selected,
        )
        self.assertTrue(
            torch.equal(
                ds[1]["image"][:3],
                torch.from_numpy(self.frames[10]).permute(2, 0, 1).float() / 255.0,
            )
        )

    def test_non_strict_stacking_preserves_length(self):
        ds = DMCDataset(self.frames, self.physics, self.episode_ids, frame_stack=2)
        self.assertEqual(len(ds), 20)
        self.assertEqual(ds[1]["image"].shape, (6, 64, 64))

    def test_strict_stacking_removes_episode_boundaries(self):
        ds = DMCDataset(
            self.frames,
            self.physics,
            self.episode_ids,
            frame_stack=2,
            strict_frame_stack=True,
        )
        self.assertEqual(len(ds), 18)
        item = ds[9]  # absolute index 10, first valid pair of episode 1
        self.assertEqual(int(item["index"].item()), 11)
        self.assertEqual(int(item["episode_id"].item()), 1)
        self.assertFalse(torch.equal(item["image"][:3], item["image"][3:]))

    def test_strict_three_frame_stack(self):
        ds = DMCDataset(
            self.frames,
            self.physics,
            self.episode_ids,
            frame_stack=3,
            strict_frame_stack=True,
        )
        self.assertEqual(len(ds), 16)
        self.assertEqual(int(ds[8]["index"].item()), 12)
        self.assertEqual(ds[8]["image"].shape, (9, 64, 64))


if __name__ == "__main__":
    unittest.main()
