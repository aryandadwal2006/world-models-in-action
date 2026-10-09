"""Audit the full DMC splits and episode-stratified sample selection.

This is a cheap data-only diagnostic. It does not train models or modify caches.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from worldmodels.data.dmc_data import (
    load_dataset_npz,
    select_episode_stratified_indices,
)


TASKS = (
    ("cartpole_balance", 4),
    ("finger_spin", 6),
    ("cheetah_run", 18),
)


def episode_counts(episode_ids: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(episode_ids, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def report_split(task: str, split: str, cap: int) -> None:
    path = os.path.join("data", f"dmc_{task}_{split}.npz")
    raw = load_dataset_npz(path)
    episode_ids = raw["episode_ids"]
    states = np.asarray(raw["physics_states"], dtype=np.float64)
    count = min(cap, len(episode_ids))

    first_indices = np.arange(count, dtype=np.int64)
    spread_indices = select_episode_stratified_indices(
        episode_ids,
        count,
    )

    print(f"\n{task} / {split}")
    print(
        f"  Full split: frames={len(episode_ids)}, "
        f"episodes={len(np.unique(episode_ids))}, "
        f"episode lengths={episode_counts(episode_ids)}"
    )
    print(
        f"  First-{count} slice: "
        f"episodes={len(np.unique(episode_ids[first_indices]))}, "
        f"counts={episode_counts(episode_ids[first_indices])}"
    )
    print(
        f"  Stratified sample: frames={len(spread_indices)}, "
        f"episodes={len(np.unique(episode_ids[spread_indices]))}, "
        f"counts={episode_counts(episode_ids[spread_indices])}"
    )

    if task == "finger_spin":
        selected = states[spread_indices]
        names = (
            "proximal_qpos",
            "distal_qpos",
            "spinner_hinge_qpos",
            "proximal_qvel",
            "distal_qvel",
            "spinner_hinge_qvel",
        )
        print("  State distribution over stratified sample:")
        for index, name in enumerate(names):
            values = selected[:, index]
            q01, median, q99 = np.quantile(
                values,
                [0.01, 0.50, 0.99],
            )
            print(
                f"    {name}: std={values.std():.6g}, "
                f"min={values.min():.6g}, q01={q01:.6g}, "
                f"median={median:.6g}, q99={q99:.6g}, "
                f"max={values.max():.6g}"
            )


def main() -> None:
    for task, _state_dim in TASKS:
        report_split(task, "train", cap=3500)
        report_split(task, "val", cap=1000)


if __name__ == "__main__":
    main()
