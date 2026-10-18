"""Tests for reproducible, episode-safe CPC retrieval-window selection."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import Subset


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ch03"
    / "11_temporal_cpc.py"
)
SPEC = importlib.util.spec_from_file_location(
    "chapter3_temporal_cpc",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import CPC script at {SCRIPT_PATH}")
CPC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CPC
SPEC.loader.exec_module(CPC)


class TestCPCRetrievalWindowSelection(unittest.TestCase):
    def setUp(self):
        # Five 1,000-frame validation episodes support 40 disjoint windows each.
        self.episode_ids = np.repeat(
            np.arange(5, dtype=np.int32),
            1000,
        )
        self.frames = np.zeros(
            (len(self.episode_ids), 2, 2, 3),
            dtype=np.uint8,
        )
        self.states = np.zeros(
            (len(self.episode_ids), 4),
            dtype=np.float32,
        )
        self.dataset = CPC.SequenceDataset(
            self.frames,
            self.states,
            self.episode_ids,
            seq_len=25,
            max_sequences=None,
        )

    def test_selects_200_episode_safe_non_overlapping_retrieval_windows(self):
        subset = CPC.episode_stratified_nonoverlapping_subset(
            self.dataset,
            200,
        )
        metadata = CPC.sequence_window_metadata(
            self.dataset,
            subset,
        )

        self.assertEqual(len(subset), 200)
        self.assertFalse(metadata["overlapping_windows"])
        self.assertEqual(
            metadata["episode_counts"],
            {str(episode): 40 for episode in range(5)},
        )

        starts_by_episode = {}
        for start, episode in zip(
            metadata["start_indices"],
            metadata["episode_ids"],
        ):
            self.assertTrue(
                np.all(
                    self.episode_ids[start : start + self.dataset.seq_len]
                    == episode
                )
            )
            starts_by_episode.setdefault(episode, []).append(start)

        for starts in starts_by_episode.values():
            starts.sort()
            self.assertTrue(
                all(
                    right - left >= self.dataset.seq_len
                    for left, right in zip(starts, starts[1:])
                )
            )

    def test_fails_instead_of_silently_returning_too_few_windows(self):
        with self.assertRaisesRegex(ValueError, "only 200 are available"):
            CPC.episode_stratified_nonoverlapping_subset(
                self.dataset,
                201,
            )

    def test_metadata_rejects_overlapping_retrieval_windows(self):
        invalid_subset = Subset(self.dataset, [0, 1])
        with self.assertRaisesRegex(RuntimeError, "overlap"):
            CPC.sequence_window_metadata(
                self.dataset,
                invalid_subset,
            )


if __name__ == "__main__":
    unittest.main()
