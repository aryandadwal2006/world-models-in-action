"""Training utilities and reproducibility helpers.

Provides deterministic seeding, device discovery, and serialization helpers
for experimental artifacts across all chapters.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Sets random seeds across standard library, NumPy, and PyTorch.

    Args:
        seed: Integer random seed for deterministic execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns the primary compute device (CUDA GPU if available, else CPU).

    Returns:
        torch.device representing the target execution device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Saves a dictionary as formatted JSON, creating parent folders if needed.

    Args:
        data: Dictionary of metrics or experiment parameters to serialize.
        filepath: Target filesystem path for the .json artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """Loads a JSON file into a dictionary.

    Args:
        filepath: Path to the .json artifact.

    Returns:
        Loaded dictionary contents.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
