"""Experiment registry backed by the platform metadata catalog."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from quant_platform.data.storage.catalog import MetadataCatalog


class ExperimentRegistry:
    """Convenience API for recording experiments and metrics."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog = MetadataCatalog(catalog_path)
        self.catalog.create_all()

    def create_experiment(
        self,
        name: str,
        *,
        status: str = "created",
        model_id: int | None = None,
        dataset_id: int | None = None,
        feature_set_id: int | None = None,
        parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Register an experiment and return its id."""

        return self.catalog.insert_row(
            "experiments",
            {
                "name": name,
                "status": status,
                "model_id": model_id,
                "dataset_id": dataset_id,
                "feature_set_id": feature_set_id,
                "parameters": dict(parameters or {}),
                "metadata": dict(metadata or {}),
            },
        )

    def start_experiment(
        self,
        experiment_id: int,
        *,
        parameters: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Mark an existing experiment as running and refresh run context."""

        updated_rows = self.catalog.update_row(
            "experiments",
            experiment_id,
            {
                "status": "running",
                "parameters": dict(parameters or {}),
                "metadata": dict(metadata or {}),
            },
        )
        if updated_rows == 0:
            raise ValueError(f"experiment does not exist: {experiment_id}")

    def log_metric(
        self,
        experiment_id: int,
        name: str,
        value: float,
        *,
        step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Record a scalar metric for an experiment and return its id."""

        return self.catalog.insert_row(
            "experiment_metrics",
            {
                "experiment_id": experiment_id,
                "name": name,
                "value": value,
                "step": step,
                "metadata": dict(metadata or {}),
            },
        )

    def list_experiments(self) -> Sequence[Mapping[str, Any]]:
        """List registered experiments."""

        return self.catalog.list_rows("experiments")

    def list_metrics(self) -> Sequence[Mapping[str, Any]]:
        """List recorded experiment metrics."""

        return self.catalog.list_rows("experiment_metrics")
