"""Linear-probe evaluation utilities.

Ridge regression is solved through a float64 least-squares formulation rather
than normal equations. This avoids avoidable numerical instability when the
latent dimension becomes large in the bottleneck sweep.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute out-of-sample R^2 per target variable.

    The denominator uses the evaluation-set mean. This is the standard
    out-of-sample coefficient-of-determination convention used throughout the
    chapter.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be two-dimensional")

    numerator = np.sum((y_true - y_pred) ** 2, axis=0)
    denominator = np.sum(
        (y_true - np.mean(y_true, axis=0, keepdims=True)) ** 2,
        axis=0,
    )
    r2 = np.empty_like(numerator)
    nonconstant = denominator > 0.0
    r2[nonconstant] = 1.0 - numerator[nonconstant] / denominator[nonconstant]
    r2[~nonconstant] = np.where(numerator[~nonconstant] == 0.0, 1.0, 0.0)
    return r2


class LinearProbe:
    """Closed-form Ridge probe for frozen representations.

    The intercept is not regularized. The solution is equivalent to Ridge
    regression, but is computed through ``np.linalg.lstsq`` on an augmented
    system, which is substantially more stable than solving normal equations.
    """

    def __init__(self, alpha: float = 1e-3) -> None:
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.alpha = float(alpha)
        self.weights: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("x and y must be two-dimensional")
        if len(x) != len(y):
            raise ValueError("x and y must contain the same number of rows")
        if len(x) == 0:
            raise ValueError("Cannot fit a probe on an empty dataset")

        n, d = x.shape
        x_bias = np.concatenate([x, np.ones((n, 1), dtype=np.float64)], axis=1)

        if self.alpha > 0.0:
            reg = np.zeros((d + 1, d + 1), dtype=np.float64)
            reg[:d, :d] = np.sqrt(self.alpha) * np.eye(d, dtype=np.float64)
            augmented_x = np.vstack([x_bias, reg])
            augmented_y = np.vstack([
                y,
                np.zeros((d + 1, y.shape[1]), dtype=np.float64),
            ])
        else:
            augmented_x = x_bias
            augmented_y = y

        self.weights, *_ = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("LinearProbe must be fitted before predict")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be two-dimensional")
        x_bias = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
        return x_bias @ self.weights

    def score(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        r2_per_var = compute_r2_score(y, self.predict(x))
        return r2_per_var, float(np.mean(r2_per_var))


def extract_features(
    encoder: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frozen representations and evaluation-only targets."""
    encoder.eval()
    latents_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["physics_state"].cpu().numpy()
            z = encoder(images)
            latents_list.append(z.detach().cpu().numpy())
            targets_list.append(targets)

    if not latents_list:
        raise ValueError("Dataloader produced no batches")

    return np.concatenate(latents_list, axis=0), np.concatenate(targets_list, axis=0)


def evaluate_linear_probe(
    encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 1e-3,
) -> Dict[str, Any]:
    """Fit a Ridge probe on train features and score it on validation features."""
    x_train, y_train = extract_features(encoder, train_loader, device)
    x_val, y_val = extract_features(encoder, val_loader, device)

    probe = LinearProbe(alpha=alpha).fit(x_train, y_train)
    r2_per_var, mean_r2 = probe.score(x_val, y_val)

    return {
        "r2_per_variable": r2_per_var.tolist(),
        "mean_r2": mean_r2,
    }
