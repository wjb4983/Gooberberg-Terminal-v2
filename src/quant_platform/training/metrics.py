"""Metric helpers for minimal training runs."""

from __future__ import annotations

import torch

from quant_platform.training.schemas import TaskType


def regression_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return simple regression metrics."""

    errors = predictions - targets
    return {
        "mae": float(errors.abs().mean().item()),
        "mse": float(torch.square(errors).mean().item()),
    }


def binary_classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Return simple binary classification metrics from logits."""

    labels = (torch.sigmoid(predictions) >= 0.5).to(targets.dtype)
    return {"accuracy": float((labels == targets).float().mean().item())}


def compute_metrics(
    task_type: TaskType, predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Compute metrics for the configured task type."""

    if task_type == TaskType.REGRESSION:
        return regression_metrics(predictions, targets)
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return binary_classification_metrics(predictions, targets)
    raise ValueError(f"unsupported task type: {task_type}")
