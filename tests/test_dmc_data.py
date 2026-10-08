"""Tests for DMC dataset stacking and episode-boundary handling."""

import unittest

import numpy as np
import torch

from worldmodels.data.dmc_data import DMCDataset


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
