"""Episode-safe contiguous sequence windows for temporal learning."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from worldmodels.data.dmc_data import select_episode_stratified_indices


class SequenceDataset(Dataset):
    """Yield episode-safe contiguous sequence windows."""

    def __init__(
        self,
        frames,
        physics_states,
        episode_ids,
        seq_len: int,
        max_sequences: int | None = None,
    ):
        if seq_len < 2:
            raise ValueError(
                "seq_len must be >= 2"
            )

        if not (
            len(frames)
            == len(physics_states)
            == len(episode_ids)
        ):
            raise ValueError(
                "frames, states, and episode_ids "
                "must have equal lengths"
            )

        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids
        self.seq_len = seq_len

        self.valid_indices = []

        for start in range(
            len(frames) - seq_len + 1
        ):
            window = episode_ids[
                start : start + seq_len
            ]

            if np.all(
                window == window[0]
            ):
                self.valid_indices.append(start)

        if max_sequences is not None and len(self.valid_indices) > max_sequences:
            self.valid_indices = select_episode_stratified_indices(
                self.episode_ids,
                max_sequences,
                candidate_indices=np.asarray(self.valid_indices, dtype=np.int64),
            ).tolist()

    def __len__(self):
        return len(
            self.valid_indices
        )

    def __getitem__(self, index):
        start = self.valid_indices[index]
        end = start + self.seq_len

        images = (
            torch.from_numpy(
                self.frames[start:end]
            )
            .permute(0, 3, 1, 2)
            .float()
            / 255.0
        )

        states = torch.from_numpy(
            self.physics_states[start:end]
        ).float()

        return {
            "images": images,
            "physics_states": states,
        }
