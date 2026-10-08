"""Unit tests for dataset loading and frame stacking (worldmodels.data.dmc_data)."""

import unittest
import numpy as np
import torch

from worldmodels.data.dmc_data import DMCDataset


class TestDMCData(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.num_frames = 20
        self.frames = np.random.randint(0, 255, (self.num_frames, 64, 64, 3), dtype=np.uint8)
        self.physics = np.random.randn(self.num_frames, 4).astype(np.float32)
        self.episode_ids = np.array([0] * 10 + [1] * 10, dtype=np.int32)

    def test_single_frame_dataset(self):
        ds = DMCDataset(self.frames, self.physics, self.episode_ids, frame_stack=1)
        self.assertEqual(len(ds), self.num_frames)

        item = ds[0]
        self.assertEqual(item["image"].shape, (3, 64, 64))
        self.assertEqual(item["physics_state"].shape, (4,))
        self.assertTrue(torch.is_tensor(item["image"]))
        self.assertTrue(item["image"].max() <= 1.0)
        self.assertTrue(item["image"].min() >= 0.0)

    def test_stacked_frame_dataset(self):
        ds = DMCDataset(self.frames, self.physics, self.episode_ids, frame_stack=2)
        self.assertEqual(len(ds), self.num_frames)

        # 2 frames * 3 channels = 6 channels
        item = ds[1]
        self.assertEqual(item["image"].shape, (6, 64, 64))

    def test_frame_stack_boundary_protection(self):
        # Index 10 is the start of episode 1
        ds = DMCDataset(self.frames, self.physics, self.episode_ids, frame_stack=2)
        item = ds[10]
        self.assertEqual(item["image"].shape, (6, 64, 64))
        # At boundary, previous frame should be clamped to current frame instead of crossing episode boundary
        f1 = item["image"][:3]
        f2 = item["image"][3:]
        self.assertTrue(torch.allclose(f1, f2))


if __name__ == "__main__":
    unittest.main()
