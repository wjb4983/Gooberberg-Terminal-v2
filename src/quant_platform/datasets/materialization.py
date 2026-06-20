"""Logical dataset materialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_platform.datasets.registry import DatasetRegistry
from quant_platform.datasets.schemas import (
    DatasetDefinition,
    DatasetMaterializationPlan,
)


class DatasetMaterializer:
    """Build read plans for datasets and optionally record physical artifacts."""

    def __init__(self, registry: DatasetRegistry | None = None) -> None:
        self.registry = registry or DatasetRegistry()

    def build_plan(
        self,
        definition: DatasetDefinition | str,
        *,
        artifact_uri: str | Path | None = None,
    ) -> DatasetMaterializationPlan:
        """Create a logical read plan without requiring a physical copy."""

        dataset = self._resolve_definition(definition)
        query: dict[str, Any] = {
            "provider": dataset.provider.value,
            "data_types": [data_type.value for data_type in dataset.data_types],
            "asset_universe": dataset.asset_universe,
            "resolution": dataset.resolution,
            "date_range": dataset.date_range.model_dump(mode="json"),
            "filters": dataset.filters,
            "feature_set": (
                dataset.feature_set.model_dump(mode="json")
                if dataset.feature_set is not None
                else None
            ),
            "version": dataset.version,
        }
        return DatasetMaterializationPlan(
            dataset=dataset,
            logical=artifact_uri is None,
            query=query,
            artifact_uri=str(artifact_uri) if artifact_uri is not None else None,
        )

    def register_view(
        self,
        definition: DatasetDefinition,
        *,
        mirror_config: bool = False,
    ) -> int:
        """Register a dataset as a logical view in the metadata catalog."""

        return self.registry.register(definition, mirror_config=mirror_config)

    def _resolve_definition(
        self, definition: DatasetDefinition | str
    ) -> DatasetDefinition:
        if isinstance(definition, DatasetDefinition):
            return definition
        dataset = self.registry.get(definition)
        if dataset is None:
            msg = f"unknown dataset definition: {definition}"
            raise KeyError(msg)
        return dataset
