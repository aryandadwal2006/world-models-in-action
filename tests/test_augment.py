"""Unit tests for data augmentation pipeline (worldmodels.data.augment)."""

import unittest
import torch

from worldmodels.data.augment import (
    random_crop_shift,
    random_intensity_jitter,
    ViewPipeline,
)


class TestAugment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.channels = 3
        self.height = 64
        self.width = 64
        self.images = torch.rand(self.batch_size, self.channels, self.height, self.width)

    def test_random_crop_shift_shape_and_bounds(self):
        shifted = random_crop_shift(self.images, max_shift=3)
        self.assertEqual(shifted.shape, self.images.shape)
        self.assertTrue(torch.all(shifted >= 0.0))
        self.assertTrue(torch.all(shifted <= 1.0))

    def test_random_crop_shift_zero_shift(self):
        shifted = random_crop_shift(self.images, max_shift=0)
        self.assertTrue(torch.allclose(shifted, self.images))

    def test_random_intensity_jitter_range(self):
        jittered = random_intensity_jitter(self.images, brightness_range=0.1, contrast_range=0.1)
        self.assertEqual(jittered.shape, self.images.shape)
        self.assertTrue(torch.all(jittered >= 0.0))
        self.assertTrue(torch.all(jittered <= 1.0))

    def test_view_pipeline_generates_distinct_pairs(self):
        pipeline = ViewPipeline(max_shift=3, brightness_range=0.1, contrast_range=0.1)
        v1, v2 = pipeline(self.images)
        self.assertEqual(v1.shape, self.images.shape)
        self.assertEqual(v2.shape, self.images.shape)
        # Views should differ due to stochastic perturbations
        diff = torch.abs(v1 - v2).sum().item()
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    unittest.main()
