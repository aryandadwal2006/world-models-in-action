"""Linear probe evaluation and temporal forward prediction diagnostics.

Implements linear probe fitting (closed-form ridge regression) to measure
linear decodability of ground truth physical state variables, as well as
multi-step forward-prediction evaluation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Computes R^2 (coefficient of determination) score per column variable.

    Equation:
        R^2_k = 1 - ( sum_i (y_{i, k} - \hat{y}_{i, k})^2 ) / ( sum_i (y_{i, k} - \bar{y}_k)^2 )

    Args:
        y_true: Ground truth target array of shape (N, K).
        y_pred: Predicted target array of shape (N, K).

    Returns:
        Array of shape (K,) containing R^2 scores for each variable.
    """
    numerator = np.sum((y_true - y_pred) ** 2, axis=0)
    denominator = np.sum((y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2, axis=0)
    # Avoid zero division if variance is zero
    r2 = 1.0 - (numerator / np.maximum(denominator, 1e-8))
    return r2


class LinearProbe:
    """Linear readout probe trained on frozen latent representations.

    Uses closed-form L2-regularized least squares (Ridge regression) for fast,
    deterministic, and hyperparameter-free fitting.
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        self.alpha = alpha
        self.weights: Optional[np.ndarray] = None  # (D + 1, K)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearProbe:
        """Fits ridge regression weights W mapping x to y.

        Args:
            x: Feature matrix of shape (N, D).
            y: Target matrix of shape (N, K).

        Returns:
            self.
        """
        n, d = x.shape
        # Add bias column
        x_bias = np.concatenate([x, np.ones((n, 1), dtype=x.dtype)], axis=1)

        # Regularization matrix (do not regularize bias)
        reg = self.alpha * np.eye(d + 1, dtype=x.dtype)
        reg[-1, -1] = 0.0

        # Normal equations: (X^T X + alpha I)^(-1) X^T Y
        self.weights = np.linalg.solve(x_bias.T @ x_bias + reg, x_bias.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts targets y given features x."""
        if self.weights is None:
            raise RuntimeError("LinearProbe must be fitted before predict.")
        n = x.shape[0]
        x_bias = np.concatenate([x, np.ones((n, 1), dtype=x.dtype)], axis=1)
        return x_bias @ self.weights

    def score(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluates probe R^2 score.

        Returns:
            Tuple of (per_variable_r2, mean_r2).
        """
        y_pred = self.predict(x)
        r2_per_var = compute_r2_score(y, y_pred)
        return r2_per_var, float(np.mean(r2_per_var))


def extract_features(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Passes dataset through a frozen encoder to extract representations and targets.

    Args:
        encoder: Trained PyTorch visual encoder.
        dataloader: PyTorch DataLoader over DMCDataset.
        device: Execution compute device.

    Returns:
        Tuple of (representations_array, physics_states_array).
    """
    encoder.eval()
    latents_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["physics_state"].numpy()

            z = encoder(images)
            latents_list.append(z.cpu().numpy())
            targets_list.append(targets)

    x_all = np.concatenate(latents_list, axis=0)
    y_all = np.concatenate(targets_list, axis=0)
    return x_all, y_all


def evaluate_linear_probe(
    encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 1e-3,
) -> Dict[str, Any]:
    """Fits linear probe on train features and evaluates R^2 on validation features."""
    x_train, y_train = extract_features(encoder, train_loader, device)
    x_val, y_val = extract_features(encoder, val_loader, device)

    probe = LinearProbe(alpha=alpha).fit(x_train, y_train)
    r2_per_var, mean_r2 = probe.score(x_val, y_val)

    return {
        "r2_per_variable": r2_per_var.tolist(),
        "mean_r2": mean_r2,
    }
