"""Loss functions for self-supervised representation learning."""

from worldmodels.losses.ntxent import nt_xent_loss
from worldmodels.losses.noncontrastive import (
    barlow_twins_loss,
    simsiam_loss,
    update_target_ema,
)
from worldmodels.losses.vicreg import vicreg_loss
from worldmodels.losses.reconstruction import masked_mse_loss

__all__ = [
    "nt_xent_loss",
    "simsiam_loss",
    "update_target_ema",
    "barlow_twins_loss",
    "vicreg_loss",
    "masked_mse_loss",
]
