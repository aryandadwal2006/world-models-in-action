"""DeepMind Control Suite (DMC) data collection, caching, and dataset loaders.

Collects offscreen rendered 64x64 pixel frames alongside true physics states
(qpos, qvel) under a uniform random policy, caches datasets to compressed .npz archives,
and exposes PyTorch Dataset wrappers supporting single-frame and frame-stacking modes.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def collect_task_dataset(
    domain_name: str,
    task_name: str,
    num_frames: int = 10000,
    seed: int = 0,
    image_size: Tuple[int, int] = (64, 64),
    camera_id: int = 0,
) -> Dict[str, np.ndarray]:
    """Interacts with a DMC environment using a uniform random policy to collect rollouts.

    The environment physics state is recorded strictly for downstream linear probe
    and evaluation diagnostics, and is never exposed during unsupervised representation training.

    Args:
        domain_name: DMC domain (e.g., 'cartpole', 'finger', 'cheetah').
        task_name: DMC task (e.g., 'balance', 'spin', 'run').
        num_frames: Total number of frames to gather.
        seed: Random seed for environment initialization and action sampling.
        image_size: Target (height, width) resolution for offscreen renders.
        camera_id: Camera index to render from.

    Returns:
        Dictionary containing frames, physics states, actions, rewards, episode boundaries.
    """
    from dm_control import suite

    np.random.seed(seed)
    env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={"random": seed})
    action_spec = env.action_spec()

    frames: List[np.ndarray] = []
    physics_states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    episode_ids: List[int] = []

    current_episode = 0
    frame_count = 0

    while frame_count < num_frames:
        time_step = env.reset()
        episode_done = False

        while not episode_done and frame_count < num_frames:
            # Render camera pixels
            pixels = env.physics.render(
                height=image_size[0],
                width=image_size[1],
                camera_id=camera_id,
            )
            # Record physics state (qpos and qvel) strictly for evaluation
            phys_state = env.physics.get_state().copy()

            # Sample random action within specification limits
            action = np.random.uniform(
                action_spec.minimum,
                action_spec.maximum,
                size=action_spec.shape,
            ).astype(np.float32)

            time_step = env.step(action)
            reward = float(time_step.reward or 0.0)
            episode_done = time_step.last()

            frames.append(pixels)
            physics_states.append(phys_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(episode_done)
            episode_ids.append(current_episode)

            frame_count += 1

        current_episode += 1

    return {
        "frames": np.stack(frames, axis=0),  # (N, H, W, 3) uint8
        "physics_states": np.stack(physics_states, axis=0).astype(np.float32),  # (N, state_dim)
        "actions": np.stack(actions, axis=0).astype(np.float32),  # (N, action_dim)
        "rewards": np.array(rewards, dtype=np.float32),  # (N,)
        "dones": np.array(dones, dtype=np.bool_),  # (N,)
        "episode_ids": np.array(episode_ids, dtype=np.int32),  # (N,)
        "domain_name": np.array(domain_name),
        "task_name": np.array(task_name),
    }


def save_dataset_npz(data_dict: Dict[str, np.ndarray], filepath: str) -> None:
    """Serializes dataset to a compressed .npz archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, **data_dict)


def load_dataset_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Loads a cached dataset archive into a dictionary of NumPy arrays."""
    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


class DMCDataset(Dataset):
    """PyTorch Dataset exposing DMC rendered frames and evaluation physics states.

    Supports optional frame stacking for temporal aliasing diagnostics.
    """

    def __init__(
        self,
        frames: np.ndarray,
        physics_states: np.ndarray,
        episode_ids: Optional[np.ndarray] = None,
        frame_stack: int = 1,
    ) -> None:
        """Initializes the dataset.

        Args:
            frames: Array of shape (N, H, W, C) in uint8.
            physics_states: Ground truth physics array of shape (N, state_dim).
            episode_ids: Optional episode index array of shape (N,).
            frame_stack: Number of consecutive frames to stack (e.g. 1 or 2).
        """
        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids if episode_ids is not None else np.zeros(len(frames), dtype=np.int32)
        self.frame_stack = frame_stack

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.frame_stack == 1:
            # (H, W, C) uint8 -> (C, H, W) float32 in [0, 1]
            frame = self.frames[index]
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        else:
            # Multi-frame stack: concatenate along channel dimension
            stacked_frames = []
            cur_ep = self.episode_ids[index]
            for offset in range(self.frame_stack - 1, -1, -1):
                idx = index - offset
                # Boundary check: do not cross episode boundaries or start of array
                if idx < 0 or self.episode_ids[idx] != cur_ep:
                    idx = index
                f = self.frames[idx]
                stacked_frames.append(torch.from_numpy(f).permute(2, 0, 1).float() / 255.0)
            frame_tensor = torch.cat(stacked_frames, dim=0)

        phys_tensor = torch.from_numpy(self.physics_states[index]).float()

        return {
            "image": frame_tensor,
            "physics_state": phys_tensor,
            "index": torch.tensor(index, dtype=torch.long),
        }
