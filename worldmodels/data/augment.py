"""View generation and data augmentation pipelines for contrastive learning.

Implements spatial shifts (crop jitter) and photometric perturbations (intensity jitter)
to generate positive view pairs from sensory observations as specified in Chapter 3.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def random_crop_shift(images: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
    """Applies a random spatial translation within [-max_shift, max_shift] pixels.

    Pad-and-crop translation preserves spatial resolution without altering camera aspect ratio.

    Args:
        images: Float tensor of shape (B, C, H, W) normalized to [0, 1].
        max_shift: Maximum pixel displacement along height and width dimensions.

    Returns:
        Translated float tensor of identical shape (B, C, H, W).
    """
    if max_shift <= 0:
        return images

    batch_size, channels, height, width = images.shape
    padded = F.pad(images, (max_shift, max_shift, max_shift, max_shift), mode="replicate")

    # Sample random integer crop coordinates per batch element
    # PyTorch grid_sample or slicing
    out = torch.empty_like(images)
    for i in range(batch_size):
        dh = torch.randint(0, 2 * max_shift + 1, (1,)).item()
        dw = torch.randint(0, 2 * max_shift + 1, (1,)).item()
        out[i] = padded[i, :, dh : dh + height, dw : dw + width]

    return out


def random_intensity_jitter(
    images: torch.Tensor,
    brightness_range: float = 0.1,
    contrast_range: float = 0.1,
) -> torch.Tensor:
    """Applies multiplicative contrast scaling and additive brightness offsets.

    Args:
        images: Float tensor of shape (B, C, H, W) normalized to [0, 1].
        brightness_range: Maximum additive delta sampled from [-b, b].
        contrast_range: Multiplicative contrast factor sampled from [1 - c, 1 + c].

    Returns:
        Photometrically augmented float tensor clipped to [0, 1].
    """
    batch_size = images.shape[0]

    # Sample contrast factors per batch item
    contrast = 1.0 + (torch.rand(batch_size, 1, 1, 1, device=images.device) * 2.0 - 1.0) * contrast_range
    # Sample brightness shifts per batch item
    brightness = (torch.rand(batch_size, 1, 1, 1, device=images.device) * 2.0 - 1.0) * brightness_range

    augmented = images * contrast + brightness
    return torch.clamp(augmented, 0.0, 1.0)


class ViewPipeline:
    """Generates stochastic view pairs (view_1, view_2) for contrastive representation learning."""

    def __init__(
        self,
        max_shift: int = 3,
        brightness_range: float = 0.1,
        contrast_range: float = 0.1,
    ) -> None:
        self.max_shift = max_shift
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def augment_single(self, images: torch.Tensor) -> torch.Tensor:
        """Applies spatial shift followed by photometric jitter."""
        x = random_crop_shift(images, max_shift=self.max_shift)
        x = random_intensity_jitter(
            x,
            brightness_range=self.brightness_range,
            contrast_range=self.contrast_range,
        )
        return x

    def __call__(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produces two distinct stochastic views of the input tensor."""
        view_1 = self.augment_single(images)
        view_2 = self.augment_single(images)
        return view_1, view_2
