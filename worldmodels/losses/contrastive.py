"""Backward-compatible imports for the split self-supervised objectives.

New code should import each objective from its concept-specific module.
The implementations live only in those modules.
"""

from worldmodels.losses.ntxent import nt_xent_loss
from worldmodels.losses.noncontrastive import (
    barlow_twins_loss,
    simsiam_loss,
    update_target_ema,
)
from worldmodels.losses.vicreg import vicreg_loss

__all__ = [
    "nt_xent_loss",
    "simsiam_loss",
    "update_target_ema",
    "barlow_twins_loss",
    "vicreg_loss",
]
