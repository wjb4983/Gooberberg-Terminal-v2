"""Loss function helpers for training."""

from __future__ import annotations

from torch import nn

from quant_platform.training.schemas import LossName


def build_loss(name: LossName) -> nn.Module:
    """Build a PyTorch loss module from a configured loss name."""

    if name == LossName.MSE:
        return nn.MSELoss()
    if name == LossName.MAE:
        return nn.L1Loss()
    if name == LossName.BCE_WITH_LOGITS:
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"unsupported loss function: {name}")
