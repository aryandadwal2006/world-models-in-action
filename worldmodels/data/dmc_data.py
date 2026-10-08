"""DeepMind Control Suite data collection, caching, and dataset utilities.

The cached target state is explicitly qpos followed by qvel. It is used only for
held-out evaluation and never enters self-supervised representation training.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _package_version(package: str) -> str:
    """Return an installed package version, or ``"unknown"`` if unavailable."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _qpos_qvel_state(physics) -> np.ndarray:
    """Return exactly ``[qpos, qvel]`` from MuJoCo physics.

    ``Physics.get_state()`` may include additional simulator state such as
    actuator activations or plugin state. The chapter's privileged target is
    deliberately restricted to generalized configuration and velocity.
    """
    qpos = np.asarray(physics.data.qpos, dtype=np.float32).copy()
    qvel = np.asarray(physics.data.qvel, dtype=np.float32).copy()
    return np.concatenate([qpos, qvel], axis=0)


def collect_task_dataset(
    domain_name: str,
    task_name: str,
    num_frames: int = 10000,
    seed: int = 0,
    image_size: Tuple[int, int] = (64, 64),
    camera_id: int = 0,
    split: str = "unspecified",
) -> Dict[str, np.ndarray]:
    """Collect rendered frames and privileged qpos/qvel targets.

    The action policy is uniform over the task action bounds. Train and
    validation sets should be collected by separate calls with different seeds,
    which guarantees that they do not share an episode.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if image_size[0] <= 0 or image_size[1] <= 0:
        raise ValueError("image_size must contain positive dimensions")

    from dm_control import suite

    rng = np.random.default_rng(seed)
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
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
            pixels = env.physics.render(
                height=image_size[0],
                width=image_size[1],
                camera_id=camera_id,
            )
            state = _qpos_qvel_state(env.physics)

            action = rng.uniform(
                action_spec.minimum,
                action_spec.maximum,
                size=action_spec.shape,
            ).astype(np.float32)

            time_step = env.step(action)
            reward = float(time_step.reward or 0.0)
            episode_done = time_step.last()

            frames.append(pixels)
            physics_states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(episode_done)
            episode_ids.append(current_episode)
            frame_count += 1

        current_episode += 1

    metadata = {
        "domain_name": domain_name,
        "task_name": task_name,
        "split": split,
        "seed": int(seed),
        "num_frames": int(frame_count),
        "image_height": int(image_size[0]),
        "image_width": int(image_size[1]),
        "camera_id": int(camera_id),
        "policy": "uniform_action_bounds",
        "state_layout": "qpos_then_qvel",
        "state_dim": int(physics_states[0].shape[0]),
        "dm_control_version": _package_version("dm-control"),
        "mujoco_version": _package_version("mujoco"),
    }

    return {
        "frames": np.stack(frames, axis=0),
        "physics_states": np.stack(physics_states, axis=0).astype(np.float32),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.bool_),
        "episode_ids": np.asarray(episode_ids, dtype=np.int32),
        "domain_name": np.asarray(domain_name),
        "task_name": np.asarray(task_name),
        "split": np.asarray(split),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }


def save_dataset_npz(data_dict: Dict[str, np.ndarray], filepath: str) -> None:
    """Serialize a dataset dictionary to a compressed NPZ archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(filepath, **data_dict)


def load_dataset_npz(filepath: str) -> Dict[str, np.ndarray]:
    """Load a cached NPZ archive."""
    with np.load(filepath, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def metadata_from_dataset(data_dict: Dict[str, np.ndarray]) -> Dict[str, object]:
    """Decode the JSON metadata stored in a dataset, if present."""
    raw = data_dict.get("metadata_json")
    if raw is None:
        return {}
    value = raw.item() if np.ndim(raw) == 0 else raw
    return json.loads(str(value))


class DMCDataset(Dataset):
    """Dataset exposing frames and evaluation-only physics states.

    ``strict_frame_stack=True`` removes indices whose requested temporal window
    would cross an episode boundary. The target is always the final frame in
    the stack.
    """

    def __init__(
        self,
        frames: np.ndarray,
        physics_states: np.ndarray,
        episode_ids: Optional[np.ndarray] = None,
        frame_stack: int = 1,
        strict_frame_stack: bool = False,
    ) -> None:
        if frame_stack < 1:
            raise ValueError("frame_stack must be >= 1")
        if len(frames) != len(physics_states):
            raise ValueError("frames and physics_states must have equal length")
        if episode_ids is None:
            episode_ids = np.zeros(len(frames), dtype=np.int32)
        if len(episode_ids) != len(frames):
            raise ValueError("episode_ids must have the same length as frames")

        self.frames = frames
        self.physics_states = physics_states
        self.episode_ids = episode_ids
        self.frame_stack = frame_stack
        self.strict_frame_stack = strict_frame_stack

        if strict_frame_stack and frame_stack > 1:
            valid_indices = []
            for index in range(frame_stack - 1, len(frames)):
                start = index - frame_stack + 1
                window_episodes = self.episode_ids[start : index + 1]
                if np.all(window_episodes == window_episodes[-1]):
                    valid_indices.append(index)
            self._indices = np.asarray(valid_indices, dtype=np.int64)
        else:
            self._indices = None

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else len(self.frames)

    def _resolve_index(self, index: int) -> int:
        return int(self._indices[index]) if self._indices is not None else int(index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        actual_index = self._resolve_index(index)

        if self.frame_stack == 1:
            frame = self.frames[actual_index]
            frame_tensor = (
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            )
        else:
            stacked_frames = []
            current_episode = self.episode_ids[actual_index]
            for offset in range(self.frame_stack - 1, -1, -1):
                candidate = actual_index - offset
                if candidate < 0 or self.episode_ids[candidate] != current_episode:
                    if self.strict_frame_stack:
                        raise RuntimeError("Strict frame stack produced an invalid window")
                    candidate = actual_index
                frame = self.frames[candidate]
                stacked_frames.append(
                    torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                )
            frame_tensor = torch.cat(stacked_frames, dim=0)

        phys_tensor = torch.from_numpy(self.physics_states[actual_index]).float()

        return {
            "image": frame_tensor,
            "physics_state": phys_tensor,
            "index": torch.tensor(actual_index, dtype=torch.long),
            "episode_id": torch.tensor(int(self.episode_ids[actual_index]), dtype=torch.long),
        }
