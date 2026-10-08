"""Linear-probe evaluation utilities for Chapter 3.

The probe standardizes frozen latent features before Ridge fitting. This keeps
regularization comparable across latent coordinates and latent dimensions.

StateLinearProbe additionally handles angular position variables by fitting
sin/cos targets and scoring them with a circular R2-like score. This avoids
artificially large errors when a physical angle crosses the -pi/pi boundary.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# Canonical qpos/qvel layouts used by Chapter 3:
#
# cartpole:
#   [slider, hinge, slider_vel, hinge_vel]
#
# finger:
#   [proximal, distal, hinge, proximal_vel, distal_vel, hinge_vel]
#
# cheetah:
#   [rootx, rootz, rooty, 6 hinge qpos, 3 root/joint velocity entries
#    corresponding to the six hinge velocities]
#
# Angular position indices refer only to qpos positions, never qvel.
CANONICAL_ANGULAR_POSITION_INDICES = {
    4: (1,),
    6: (2,),
    18: tuple(range(2, 9)),
}

# Periods are properties of the rendered state variables, not merely of MuJoCo
# joint types. The finger spinner has two identical, opposite caps and hidden
# tip markers, so its rendered geometry is invariant under theta -> theta + pi.
CANONICAL_ANGULAR_POSITION_PERIODS = {
    4: {1: 2.0 * np.pi},
    6: {2: np.pi},
    18: {index: 2.0 * np.pi for index in range(2, 9)},
}


def infer_angular_position_indices(
    state_dim: int,
) -> Tuple[int, ...]:
    """Return canonical angular-position indices for Chapter 3 tasks."""
    try:
        return tuple(CANONICAL_ANGULAR_POSITION_INDICES[state_dim])
    except KeyError as exc:
        raise ValueError(
            f"No canonical angular-position mapping exists for "
            f"state_dim={state_dim}; pass "
            "angular_position_indices explicitly"
        ) from exc


def infer_angular_position_periods(
    state_dim: int,
) -> Dict[int, float]:
    """Return the rendered angular period for each canonical angle."""
    try:
        return dict(CANONICAL_ANGULAR_POSITION_PERIODS[state_dim])
    except KeyError as exc:
        raise ValueError(
            f"No canonical angular-period mapping exists for "
            f"state_dim={state_dim}"
        ) from exc


def _normalize_angular_periods(
    indices: Sequence[int],
    angular_position_periods: Optional[Dict[int, float]] = None,
    defaults: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """Validate periods and supply 2*pi for unspecified angles."""
    supplied = {
        int(index): float(period)
        for index, period in (angular_position_periods or {}).items()
    }
    defaults = defaults or {}
    result = {}
    for index in indices:
        period = supplied.get(index, defaults.get(index, 2.0 * np.pi))
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("Every angular period must be finite and positive")
        result[index] = period
    return result


def _validate_angular_indices(
    state_dim: int,
    angular_position_indices: Sequence[int],
) -> Tuple[int, ...]:
    """Validate and normalize angular state-variable indices."""
    indices = tuple(
        sorted(
            set(
                int(index)
                for index in angular_position_indices
            )
        )
    )

    if any(
        index < 0 or index >= state_dim
        for index in indices
    ):
        raise ValueError(
            "angular_position_indices contains "
            "an out-of-range index"
        )

    return indices


def expanded_state_dim(
    state_dim: int,
    angular_position_indices: Sequence[int],
) -> int:
    """Return output dimension after replacing angles by sin/cos."""
    indices = _validate_angular_indices(
        state_dim,
        angular_position_indices,
    )

    return state_dim + len(indices)


def transform_state_targets_np(
    states: np.ndarray,
    angular_position_indices: Sequence[int],
    angular_position_periods: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Transform raw state targets to linear-probe targets.

    Ordinary variables remain unchanged. Each angular variable theta becomes
    two targets: sin(theta), cos(theta).
    """
    states = np.asarray(
        states,
        dtype=np.float64,
    )

    if states.ndim != 2:
        raise ValueError(
            "states must be two-dimensional"
        )

    indices = _validate_angular_indices(
        states.shape[1],
        angular_position_indices,
    )

    try:
        canonical_periods = infer_angular_position_periods(states.shape[1])
    except ValueError:
        canonical_periods = {}
    periods = _normalize_angular_periods(
        indices,
        angular_position_periods,
        canonical_periods,
    )
    angular_set = set(indices)
    columns = []

    for index in range(states.shape[1]):
        if index in angular_set:
            phase = (2.0 * np.pi / periods[index]) * states[:, index]

            columns.append(
                np.sin(phase)[:, None]
            )
            columns.append(
                np.cos(phase)[:, None]
            )
        else:
            columns.append(
                states[:, index : index + 1]
            )

    return np.concatenate(
        columns,
        axis=1,
    )


def transform_state_targets_torch(
    states: torch.Tensor,
    angular_position_indices: Sequence[int],
    angular_position_periods: Optional[Dict[int, float]] = None,
) -> torch.Tensor:
    """Torch equivalent of ``transform_state_targets_np``."""
    if states.ndim != 2:
        raise ValueError(
            "states must be two-dimensional"
        )

    indices = _validate_angular_indices(
        states.shape[1],
        angular_position_indices,
    )

    try:
        canonical_periods = infer_angular_position_periods(states.shape[1])
    except ValueError:
        canonical_periods = {}
    periods = _normalize_angular_periods(
        indices,
        angular_position_periods,
        canonical_periods,
    )
    angular_set = set(indices)
    columns = []

    for index in range(states.shape[1]):
        if index in angular_set:
            phase = (2.0 * np.pi / periods[index]) * states[:, index : index + 1]

            columns.append(
                torch.sin(phase)
            )
            columns.append(
                torch.cos(phase)
            )
        else:
            columns.append(
                states[:, index : index + 1]
            )

    return torch.cat(
        columns,
        dim=1,
    )


def _decode_transformed_state_targets(
    transformed: np.ndarray,
    state_dim: int,
    angular_position_indices: Sequence[int],
    angular_position_periods: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Decode sin/cos target pairs back into canonical angles."""
    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    if transformed.ndim != 2:
        raise ValueError(
            "transformed must be two-dimensional"
        )

    indices = _validate_angular_indices(
        state_dim,
        angular_position_indices,
    )

    expected_dim = expanded_state_dim(
        state_dim,
        indices,
    )

    if transformed.shape[1] != expected_dim:
        raise ValueError(
            f"Expected transformed target dimension "
            f"{expected_dim}, got {transformed.shape[1]}"
        )

    try:
        canonical_periods = infer_angular_position_periods(state_dim)
    except ValueError:
        canonical_periods = {}
    periods = _normalize_angular_periods(
        indices,
        angular_position_periods,
        canonical_periods,
    )
    angular_set = set(indices)

    output = np.empty(
        (len(transformed), state_dim),
        dtype=np.float64,
    )

    cursor = 0

    for index in range(state_dim):
        if index in angular_set:
            sin_theta = transformed[:, cursor]
            cos_theta = transformed[:, cursor + 1]

            output[:, index] = (
                np.arctan2(sin_theta, cos_theta)
                * periods[index]
                / (2.0 * np.pi)
            )

            cursor += 2
        else:
            output[:, index] = transformed[:, cursor]
            cursor += 1

    return output


def compute_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Compute ordinary out-of-sample R2 per target variable."""
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes"
        )

    if y_true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be two-dimensional"
        )

    numerator = np.sum(
        (y_true - y_pred) ** 2,
        axis=0,
    )

    denominator = np.sum(
        (
            y_true
            - np.mean(
                y_true,
                axis=0,
                keepdims=True,
            )
        ) ** 2,
        axis=0,
    )

    r2 = np.empty_like(
        numerator
    )

    nonconstant = denominator > 0.0

    r2[nonconstant] = (
        1.0
        - numerator[nonconstant]
        / denominator[nonconstant]
    )

    r2[~nonconstant] = np.where(
        numerator[~nonconstant] == 0.0,
        1.0,
        0.0,
    )

    return r2


def circular_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    period: float = 2.0 * np.pi,
) -> float:
    """Compute an R2-like score using squared chordal distance.

    For angles theta:

        d^2(theta_a, theta_b)
            = 2 - 2 cos(theta_a - theta_b)

    The baseline prediction is the circular-mean direction. The score is
    invariant to adding any multiple of 2*pi to either angle.
    """
    true = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    pred = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    if true.shape != pred.shape:
        raise ValueError(
            "Angular targets must have identical shapes"
        )

    if true.size == 0:
        raise ValueError(
            "Angular targets cannot be empty"
        )

    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be finite and positive")

    scale = 2.0 * np.pi / period
    true_phase = scale * true
    pred_phase = scale * pred
    resultant = np.mean(
        np.column_stack(
            [
                np.sin(true_phase),
                np.cos(true_phase),
            ]
        ),
        axis=0,
    )

    mean_phase = float(
        np.arctan2(
            resultant[0],
            resultant[1],
        )
    )

    numerator = float(
        np.sum(
            2.0
            - 2.0 * np.cos(pred_phase - true_phase)
        )
    )

    denominator = float(
        np.sum(
            2.0
            - 2.0 * np.cos(true_phase - mean_phase)
        )
    )

    if denominator <= 1e-12:
        return (
            1.0
            if numerator <= 1e-12
            else 0.0
        )

    return 1.0 - numerator / denominator


def compute_structured_r2_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    angular_position_indices: Sequence[int],
    angular_position_periods: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Compute one R2 score per original physical-state variable.

    Angular variables use circular_r2_score. All other variables use ordinary
    out-of-sample R2.
    """
    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=np.float64,
    )

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have identical shapes"
        )

    if y_true.ndim != 2:
        raise ValueError(
            "y_true and y_pred must be two-dimensional"
        )

    indices = _validate_angular_indices(
        y_true.shape[1],
        angular_position_indices,
    )

    try:
        canonical_periods = infer_angular_position_periods(y_true.shape[1])
    except ValueError:
        canonical_periods = {}
    periods = _normalize_angular_periods(
        indices,
        angular_position_periods,
        canonical_periods,
    )
    angular_set = set(indices)

    result = np.empty(
        y_true.shape[1],
        dtype=np.float64,
    )

    ordinary_indices = [
        index
        for index in range(y_true.shape[1])
        if index not in angular_set
    ]

    if ordinary_indices:
        ordinary_r2 = compute_r2_score(
            y_true[:, ordinary_indices],
            y_pred[:, ordinary_indices],
        )

        result[ordinary_indices] = ordinary_r2

    for index in indices:
        result[index] = circular_r2_score(
            y_true[:, index],
            y_pred[:, index],
            period=periods[index],
        )

    return result


class LinearProbe:
    """Closed-form Ridge probe with standardized input features."""

    def __init__(
        self,
        alpha: float = 1e-3,
    ) -> None:
        if alpha < 0:
            raise ValueError(
                "alpha must be non-negative"
            )

        self.alpha = float(alpha)
        self.weights: Optional[np.ndarray] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def _fit_feature_scaler(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        self.feature_mean = np.mean(
            x,
            axis=0,
        )

        self.feature_std = np.std(
            x,
            axis=0,
        )

        self.feature_std = np.where(
            self.feature_std > 1e-12,
            self.feature_std,
            1.0,
        )

        return (
            x - self.feature_mean
        ) / self.feature_std

    def _transform_features(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if (
            self.feature_mean is None
            or self.feature_std is None
        ):
            raise RuntimeError(
                "LinearProbe must be fitted before predict"
            )

        x = np.asarray(
            x,
            dtype=np.float64,
        )

        if x.ndim != 2:
            raise ValueError(
                "x must be two-dimensional"
            )

        if x.shape[1] != len(
            self.feature_mean
        ):
            raise ValueError(
                "x has a different number of "
                "features from the fitted probe"
            )

        return (
            x - self.feature_mean
        ) / self.feature_std

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "LinearProbe":
        x = np.asarray(
            x,
            dtype=np.float64,
        )

        y = np.asarray(
            y,
            dtype=np.float64,
        )

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(
                "x and y must be two-dimensional"
            )

        if len(x) != len(y):
            raise ValueError(
                "x and y must contain the same number of rows"
            )

        if len(x) == 0:
            raise ValueError(
                "Cannot fit a probe on an empty dataset"
            )

        x_scaled = self._fit_feature_scaler(x)

        n, d = x_scaled.shape

        x_bias = np.concatenate(
            [
                x_scaled,
                np.ones(
                    (n, 1),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        if self.alpha > 0.0:
            reg = np.zeros(
                (d + 1, d + 1),
                dtype=np.float64,
            )

            reg[:d, :d] = (
                np.sqrt(self.alpha)
                * np.eye(
                    d,
                    dtype=np.float64,
                )
            )

            augmented_x = np.vstack(
                [
                    x_bias,
                    reg,
                ]
            )

            augmented_y = np.vstack(
                [
                    y,
                    np.zeros(
                        (
                            d + 1,
                            y.shape[1],
                        ),
                        dtype=np.float64,
                    ),
                ]
            )
        else:
            augmented_x = x_bias
            augmented_y = y

        self.weights, *_ = np.linalg.lstsq(
            augmented_x,
            augmented_y,
            rcond=None,
        )

        return self

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError(
                "LinearProbe must be fitted before predict"
            )

        x_scaled = self._transform_features(
            x
        )

        x_bias = np.concatenate(
            [
                x_scaled,
                np.ones(
                    (
                        len(x_scaled),
                        1,
                    ),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        return x_bias @ self.weights

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        r2_per_var = compute_r2_score(
            y,
            self.predict(x),
        )

        return (
            r2_per_var,
            float(np.mean(r2_per_var)),
        )


class StateLinearProbe:
    """Ridge probe for physical state with circular angular positions."""

    def __init__(
        self,
        alpha: float = 1e-3,
        angular_position_indices: Optional[Sequence[int]] = None,
        angular_position_periods: Optional[Dict[int, float]] = None,
    ) -> None:
        self.alpha = float(alpha)

        self.angular_position_indices = (
            tuple(angular_position_indices)
            if angular_position_indices is not None
            else None
        )
        self.angular_position_periods = (
            {int(k): float(v) for k, v in angular_position_periods.items()}
            if angular_position_periods is not None
            else None
        )

        self.probe = LinearProbe(
            alpha=alpha
        )

        self.state_dim: Optional[int] = None

    def _indices(
        self,
        state_dim: int,
    ) -> Tuple[int, ...]:
        if self.angular_position_indices is None:
            return infer_angular_position_indices(
                state_dim
            )

        return _validate_angular_indices(
            state_dim,
            self.angular_position_indices,
        )

    def _periods(self, state_dim: int) -> Dict[int, float]:
        indices = self._indices(state_dim)
        if self.angular_position_periods is not None:
            return _normalize_angular_periods(
                indices,
                self.angular_position_periods,
            )
        try:
            defaults = infer_angular_position_periods(state_dim)
        except ValueError:
            defaults = {}
        return _normalize_angular_periods(indices, defaults=defaults)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> "StateLinearProbe":
        y = np.asarray(
            y,
            dtype=np.float64,
        )

        if y.ndim != 2:
            raise ValueError(
                "y must be two-dimensional"
            )

        self.state_dim = y.shape[1]

        indices = self._indices(
            self.state_dim
        )

        periods = self._periods(self.state_dim)
        y_transformed = transform_state_targets_np(
            y,
            indices,
            angular_position_periods=periods,
        )

        self.probe.fit(
            x,
            y_transformed,
        )

        return self

    def predict(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        if self.state_dim is None:
            raise RuntimeError(
                "StateLinearProbe must be fitted before predict"
            )

        indices = self._indices(
            self.state_dim
        )

        transformed = self.probe.predict(
            x
        )

        return _decode_transformed_state_targets(
            transformed,
            self.state_dim,
            indices,
            angular_position_periods=self._periods(self.state_dim),
        )

    def score(
        self,
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        if self.state_dim is None:
            raise RuntimeError(
                "StateLinearProbe must be fitted before score"
            )

        y = np.asarray(
            y,
            dtype=np.float64,
        )

        y_pred = self.predict(x)

        r2_per_variable = (
            compute_structured_r2_score(
                y,
                y_pred,
                self._indices(self.state_dim),
                angular_position_periods=self._periods(self.state_dim),
            )
        )

        return (
            r2_per_variable,
            float(
                np.mean(
                    r2_per_variable
                )
            ),
        )


class AngleAwareStateProbe(StateLinearProbe):
    """Descriptive alias for StateLinearProbe."""


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
            images = batch[
                "image"
            ].to(device)

            targets = (
                batch[
                    "physics_state"
                ]
                .cpu()
                .numpy()
            )

            z = encoder(images)

            latents_list.append(
                z.detach()
                .cpu()
                .numpy()
            )

            targets_list.append(
                targets
            )

    if not latents_list:
        raise ValueError(
            "Dataloader produced no batches"
        )

    return (
        np.concatenate(
            latents_list,
            axis=0,
        ),
        np.concatenate(
            targets_list,
            axis=0,
        ),
    )


def evaluate_linear_probe(
    encoder: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    alpha: float = 1e-3,
    angular_position_indices: Optional[
        Sequence[int]
    ] = None,
) -> Dict[str, Any]:
    """Fit an angle-aware Ridge state probe and score validation data."""
    x_train, y_train = extract_features(
        encoder,
        train_loader,
        device,
    )

    x_val, y_val = extract_features(
        encoder,
        val_loader,
        device,
    )

    probe = StateLinearProbe(
        alpha=alpha,
        angular_position_indices=angular_position_indices,
    )

    probe.fit(
        x_train,
        y_train,
    )

    r2_per_var, mean_r2 = probe.score(
        x_val,
        y_val,
    )

    if probe.state_dim is None:
        raise RuntimeError(
            "StateLinearProbe did not infer state dimension"
        )

    return {
        "r2_per_variable": r2_per_var.tolist(),
        "mean_r2": mean_r2,
        "probe": {
            "type": "StateLinearProbe",
            "feature_standardization": True,
            "angular_position_encoding": "sin_cos",
            "angular_r2": "circular_chordal",
            "angular_position_indices": list(
                probe._indices(probe.state_dim)
            ),
            "angular_position_periods": {
                str(index): float(period)
                for index, period in probe._periods(probe.state_dim).items()
            },
        },
    }