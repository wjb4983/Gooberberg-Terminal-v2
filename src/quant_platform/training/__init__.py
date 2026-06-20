"""Training utilities for quant_platform."""

from quant_platform.training.runner import run_training
from quant_platform.training.schemas import TrainingConfig

__all__ = ["TrainingConfig", "run_training"]
