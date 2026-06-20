"""Registry for reusable model definitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from quant_platform.data.storage.catalog import MetadataCatalog, model_definitions
from quant_platform.models.schemas import ModelDefinition


class ModelRegistry:
    """Persist and retrieve reusable model definitions from the metadata catalog."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        self.catalog = MetadataCatalog(catalog_path)
        self.catalog.create_all()

    def register(self, definition: ModelDefinition, *, overwrite: bool = True) -> int:
        """Persist a model definition and return its catalog id."""

        existing = self.get_row(definition.name)
        values = {
            "name": definition.name,
            "version": definition.version,
            "model_type": definition.model_type.value,
            "artifact_uri": None,
            "parameters": definition.to_parameters(),
            "metadata": definition.metadata,
        }
        if existing is not None:
            if not overwrite:
                msg = f"model definition already exists: {definition.name}"
                raise ValueError(msg)
            model_id = int(existing["id"])
            with self.catalog.engine.begin() as connection:
                connection.execute(
                    update(model_definitions)
                    .where(model_definitions.c.id == model_id)
                    .values(**values)
                )
        else:
            model_id = self.catalog.insert_row("model_definitions", values)
        return model_id

    def get_row(self, name: str) -> Mapping[str, Any] | None:
        """Return the raw catalog row for a model by name."""

        with self.catalog.engine.connect() as connection:
            row = (
                connection.execute(
                    select(model_definitions).where(model_definitions.c.name == name)
                )
                .mappings()
                .first()
            )
        return row

    def get(self, name: str) -> ModelDefinition | None:
        """Return a model definition by name, if present."""

        row = self.get_row(name)
        if row is None:
            return None
        return ModelDefinition.from_catalog_row(row)

    def list(self) -> Sequence[ModelDefinition]:
        """List registered model definitions."""

        definitions = []
        for row in self.catalog.list_rows("model_definitions"):
            definitions.append(ModelDefinition.from_catalog_row(row))
        return definitions
