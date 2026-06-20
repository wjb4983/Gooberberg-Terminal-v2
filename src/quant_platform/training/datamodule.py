"""Synthetic data module used by minimal training runs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset

from quant_platform.training.schemas import TaskType, TrainingConfig


@dataclass(frozen=True)
class DataLoaders:
    """Container for train, validation, and optional test loaders."""

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None


class SyntheticDataModule:
    """Create deterministic fake sequence data from the training configuration."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.input_dim = len(config.feature_set)

    def dataloaders(self) -> DataLoaders:
        """Build deterministic synthetic data loaders for each configured split."""

        return DataLoaders(
            train=self._loader("train"),
            validation=self._loader("validation"),
            test=self._loader("test") if self.config.date_split.test_start else None,
        )

    def _loader(self, split: str) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        dates = self.config.date_split
        if split == "train":
            days = (dates.train_end - dates.train_start).days + 1
            offset = 0
        elif split == "validation":
            days = (dates.validation_end - dates.validation_start).days + 1
            offset = 1_000
        elif (
            split == "test"
            and dates.test_start is not None
            and dates.test_end is not None
        ):
            days = (dates.test_end - dates.test_start).days + 1
            offset = 2_000
        else:
            raise ValueError(f"unsupported or incomplete split: {split}")

        row_count = max(1, days * self.config.synthetic_rows_per_day)
        generator = torch.Generator().manual_seed(self.config.seed + offset)
        features = torch.randn(
            row_count,
            self.config.sequence_length,
            self.input_dim,
            generator=generator,
        )
        weights = torch.linspace(0.5, 1.5, steps=self.input_dim).view(1, 1, -1)
        target = (features[:, -1:, :] * weights).sum(dim=2)
        target = target / float(self.input_dim)
        if self.config.task_type == TaskType.BINARY_CLASSIFICATION:
            target = (target > target.median()).float()
        dataset = TensorDataset(features.float(), target.float())
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=split == "train"
        )
