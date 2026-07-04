"""Data modules used by training runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from quant_platform.training.schemas import TaskType, TrainingConfig


@dataclass(frozen=True)
class DataLoaders:
    """Container for train, validation, and optional test loaders."""

    train: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    validation: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    test: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None


class MarketDataModule:
    """Create deterministic sequence data from ingested market-data parquet files."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.input_dim = len(config.feature_set)

    def dataloaders(self) -> DataLoaders:
        frame = self._load_frame()
        return DataLoaders(
            train=self._loader(frame, "train"),
            validation=self._loader(frame, "validation"),
            test=self._loader(frame, "test")
            if self.config.date_split.test_start
            else None,
        )

    def _load_frame(self) -> pd.DataFrame:
        root = Path(self.config.data_lake_root)
        files = sorted(root.glob("market_data/**/symbol=*/**/*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No ingested market-data parquet files found under {root}. "
                "Run dataset ingestion before training this dataset."
            )
        frames = []
        for file in files:
            frame = pd.read_parquet(file)
            symbol = next(
                (
                    part.split("=", 1)[1]
                    for part in file.parts
                    if part.startswith("symbol=")
                ),
                None,
            )
            if symbol and "symbol" not in frame.columns:
                frame["symbol"] = symbol
            frames.append(frame)
        data = pd.concat(frames, ignore_index=True)
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(
                data["timestamp"], utc=True, errors="coerce"
            )
        elif "datetime" in data.columns:
            data["timestamp"] = pd.to_datetime(
                data["datetime"], utc=True, errors="coerce"
            )
        elif "date" in data.columns:
            data["timestamp"] = pd.to_datetime(data["date"], utc=True, errors="coerce")
        else:
            raise ValueError(
                "Market-data parquet files must include timestamp, datetime, "
                "or date column"
            )
        data = data.dropna(subset=["timestamp"]).sort_values("timestamp")
        start = pd.Timestamp(self.config.date_split.train_start, tz="UTC")
        end = pd.Timestamp(
            self.config.date_split.validation_end, tz="UTC"
        ) + pd.Timedelta(days=1)
        data = data[(data["timestamp"] >= start) & (data["timestamp"] < end)]
        if data.empty:
            raise ValueError(
                "No market-data rows overlap the configured training/validation "
                "date split"
            )
        if "close" not in data.columns:
            for candidate in ("price", "last", "open"):
                if candidate in data.columns:
                    data["close"] = pd.to_numeric(data[candidate], errors="coerce")
                    break
        if "close" not in data.columns:
            raise ValueError(
                "Market-data rows must include a close column or price-like fallback"
            )
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data = data.dropna(subset=["close"])
        data["return"] = (
            data.groupby("symbol")["close"].pct_change().fillna(0.0)
            if "symbol" in data.columns
            else data["close"].pct_change().fillna(0.0)
        )
        for column in self.config.feature_set:
            if column not in data.columns:
                if column in {"feature_0", "return"}:
                    data[column] = data["return"]
                elif column in {"feature_1", "close"}:
                    data[column] = data["close"].pct_change().fillna(0.0)
                elif column in {"feature_2", "volume"} and "volume" in data.columns:
                    data[column] = (
                        pd.to_numeric(data["volume"], errors="coerce")
                        .pct_change()
                        .fillna(0.0)
                    )
                else:
                    data[column] = 0.0
        data["target"] = (
            data.groupby("symbol")["return"].shift(-self.config.target.horizon)
            if "symbol" in data.columns
            else data["return"].shift(-self.config.target.horizon)
        )
        data = data.dropna(subset=["target", *self.config.feature_set])
        if len(data) <= self.config.sequence_length:
            raise ValueError("Not enough market-data rows to build training sequences")
        return data

    def _loader(
        self, frame: pd.DataFrame, split: str
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        dates = self.config.date_split
        if split == "train":
            start, end = dates.train_start, dates.train_end
        elif split == "validation":
            start, end = dates.validation_start, dates.validation_end
        elif (
            split == "test"
            and dates.test_start is not None
            and dates.test_end is not None
        ):
            start, end = dates.test_start, dates.test_end
        else:
            raise ValueError(f"unsupported or incomplete split: {split}")
        mask = (frame["timestamp"] >= pd.Timestamp(start, tz="UTC")) & (
            frame["timestamp"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
        )
        split_frame = frame.loc[mask].sort_values("timestamp")
        features, targets = [], []
        values = split_frame[self.config.feature_set].astype("float32").to_numpy()
        target_values = split_frame["target"].astype("float32").to_numpy()
        for idx in range(self.config.sequence_length - 1, len(split_frame)):
            features.append(values[idx - self.config.sequence_length + 1 : idx + 1])
            targets.append([target_values[idx]])
        if not features:
            raise ValueError(f"Not enough {split} rows to build sequences")
        dataset = TensorDataset(
            torch.from_numpy(np.asarray(features, dtype="float32")),
            torch.from_numpy(np.asarray(targets, dtype="float32")),
        )
        return DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=split == "train"
        )


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
