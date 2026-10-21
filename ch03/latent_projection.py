"""Publication-oriented PCA plot for before/after encoder features.

Both panels use one pooled PCA basis and shared axis limits so their geometry
can be compared. Color encodes cart position; marker shape encodes wrapped pole
angle. This is a qualitative projection, not a test of linear decodability.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


ANGLE_BINS = (
    ("Upright", "o", lambda a: np.abs(a) <= np.pi / 4),
    ("Side, positive", "s", lambda a: (a > np.pi / 4) & (a <= 3 * np.pi / 4)),
    ("Hanging", "^", lambda a: np.abs(a) > 3 * np.pi / 4),
    ("Side, negative", "v", lambda a: (a >= -3 * np.pi / 4) & (a < -np.pi / 4)),
)


def _shared_pca_projection(
    before: np.ndarray,
    after: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one PCA basis to the pooled features and project both conditions."""
    before = np.asarray(before, dtype=np.float64)
    after = np.asarray(after, dtype=np.float64)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("PCA features must be two-dimensional")
    if before.shape[1] != after.shape[1]:
        raise ValueError("Before/after features must have matching dimensions")
    if before.shape[0] < 2 or after.shape[0] < 2:
        raise ValueError("Each condition must contain at least two features")
    pooled = np.concatenate([before, after], axis=0)
    center = pooled.mean(axis=0, keepdims=True)
    centered = pooled - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    return (before - center) @ components, (after - center) @ components


def plot_latent_projection(
    z_untrained: np.ndarray,
    z_trained: np.ndarray,
    physics_states: np.ndarray,
    filepath: str | Path,
) -> None:
    """Save a shared-axis, grayscale-safe PCA projection for the same samples."""
    z_untrained = np.asarray(z_untrained)
    z_trained = np.asarray(z_trained)
    states = np.asarray(physics_states)
    if states.ndim != 2 or states.shape[1] < 2:
        raise ValueError("physics_states must include cart position and pole angle")
    if len(z_untrained) != len(z_trained) or len(z_trained) != len(states):
        raise ValueError("Features and physical-state targets must have equal lengths")
    if not np.isfinite(z_untrained).all() or not np.isfinite(z_trained).all():
        raise ValueError("Features must be finite")

    before_2d, after_2d = _shared_pca_projection(z_untrained, z_trained)
    cart_position = states[:, 0]
    theta = (states[:, 1] + np.pi) % (2.0 * np.pi) - np.pi
    norm = Normalize(vmin=float(np.min(cart_position)), vmax=float(np.max(cart_position)))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharex=True, sharey=True)
    for ax, projection, title in (
        (axes[0], before_2d, "Before training"),
        (axes[1], after_2d, "After NT-Xent training"),
    ):
        for label, marker, membership in ANGLE_BINS:
            mask = membership(theta)
            if not np.any(mask):
                continue
            ax.scatter(
                projection[mask, 0],
                projection[mask, 1],
                c=cart_position[mask],
                cmap="Greys",
                norm=norm,
                marker=marker,
                s=19,
                alpha=0.72,
                edgecolors="#303030",
                linewidths=0.25,
                rasterized=True,
            )
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel("Pooled principal component 1", fontsize=9)
        ax.grid(True, linestyle=":", linewidth=0.6, color="#b0b0b0", alpha=0.55)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Pooled principal component 2", fontsize=9)

    all_projection = np.concatenate([before_2d, after_2d], axis=0)
    mins = np.min(all_projection, axis=0)
    maxs = np.max(all_projection, axis=0)
    padding = np.maximum((maxs - mins) * 0.06, 1e-4)
    axes[0].set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
    axes[0].set_ylim(mins[1] - padding[1], maxs[1] + padding[1])

    marker_handles = [
        Line2D(
            [0], [0], marker=marker, color="white",
            markerfacecolor="#7a7a7a", markeredgecolor="#303030",
            markersize=7, linestyle="None", label=label,
        )
        for label, marker, _membership in ANGLE_BINS
    ]
    fig.legend(
        handles=marker_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.45, 0.035),
    )
    scalar_map = ScalarMappable(norm=norm, cmap="Greys")
    scalar_map.set_array([])
    colorbar = fig.colorbar(scalar_map, ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label("Cart position (lighter → darker)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Shared PCA basis and axis scales; marker shape shows wrapped pole angle",
        fontsize=10,
        y=0.99,
    )
    fig.subplots_adjust(left=0.08, right=0.90, top=0.86, bottom=0.20, wspace=0.12)

    output = Path(filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
