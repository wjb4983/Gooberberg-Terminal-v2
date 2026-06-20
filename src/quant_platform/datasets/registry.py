"""Registry for logical dataset definitions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from sqlalchemy import select, update

from quant_platform.data.storage.catalog import MetadataCatalog, dataset_definitions
from quant_platform.datasets.schemas import DatasetDefinition

DEFAULT_DATASET_CONFIG_DIR = Path("configs/datasets")


class DatasetRegistry:
    """Persist and retrieve logical dataset definitions from the metadata catalog."""

    def __init__(
        self,
        catalog_path: str | Path | None = None,
        *,
        config_dir: str | Path | None = DEFAULT_DATASET_CONFIG_DIR,
    ) -> None:
        self.catalog = MetadataCatalog(catalog_path)
        self.catalog.create_all()
        self.config_dir = Path(config_dir) if config_dir is not None else None

    def register(
        self,
        definition: DatasetDefinition,
        *,
        mirror_config: bool = False,
        config_format: Literal["json", "yaml"] = "json",
        overwrite: bool = True,
    ) -> int:
        """Persist a logical dataset definition and optionally mirror a config file."""

        existing = self.get_row(definition.name)
        values = {
            "name": definition.name,
            "version": definition.version,
            "description": definition.description,
            "schema": definition.to_catalog_schema(),
            "storage_uri": None,
            "metadata": definition.metadata,
        }
        if existing is not None:
            if not overwrite:
                msg = f"dataset definition already exists: {definition.name}"
                raise ValueError(msg)
            dataset_id = int(existing["id"])
            with self.catalog.engine.begin() as connection:
                connection.execute(
                    update(dataset_definitions)
                    .where(dataset_definitions.c.id == dataset_id)
                    .values(**values)
                )
        else:
            dataset_id = self.catalog.insert_row("dataset_definitions", values)

        if mirror_config:
            self.mirror_config(definition, config_format=config_format)
        return dataset_id

    def get_row(self, name: str) -> Mapping[str, Any] | None:
        """Return the raw catalog row for a dataset by name."""

        with self.catalog.engine.connect() as connection:
            row = connection.execute(
                select(dataset_definitions).where(dataset_definitions.c.name == name)
            ).mappings().first()
        return row

    def get(self, name: str) -> DatasetDefinition | None:
        """Return a dataset definition by name, if present."""

        row = self.get_row(name)
        if row is None:
            return None
        payload = dict(row["schema"])
        payload["description"] = row["description"]
        payload["metadata"] = row["metadata"] or {}
        return DatasetDefinition(**payload)

    def list(self) -> Sequence[DatasetDefinition]:
        """List registered logical dataset definitions."""

        definitions = []
        for row in self.catalog.list_rows("dataset_definitions"):
            definition = self.get(str(row["name"]))
            if definition is not None:
                definitions.append(definition)
        return definitions

    def mirror_config(
        self,
        definition: DatasetDefinition,
        *,
        config_format: Literal["json", "yaml"] = "json",
    ) -> Path:
        """Write the dataset definition to configs/datasets as JSON or YAML."""

        if self.config_dir is None:
            msg = "config mirroring is disabled for this registry"
            raise ValueError(msg)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        suffix = "yaml" if config_format == "yaml" else "json"
        path = self.config_dir / f"{definition.name}.{suffix}"
        data = definition.to_config_dict()
        if config_format == "yaml":
            path.write_text(_to_yaml(data), encoding="utf-8")
        else:
            content = json.dumps(data, indent=2, sort_keys=True) + "\n"
            path.write_text(content, encoding="utf-8")
        return path


def _to_yaml(value: Mapping[str, Any]) -> str:
    """Serialize simple JSON-compatible data to YAML without requiring PyYAML."""

    try:
        import yaml
    except ImportError:
        return _simple_yaml(value)
    return str(yaml.safe_dump(dict(value), sort_keys=True))


def _simple_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, Mapping):
        lines = []
        for key, item in value.items():
            if isinstance(item, Mapping | list):
                lines.append(f"{prefix}{key}:")
                lines.append(_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}{key}: {json.dumps(item)}")
        return "\n".join(lines) + "\n"
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, Mapping | list):
                lines.append(f"{prefix}-")
                lines.append(_simple_yaml(item, indent + 2).rstrip())
            else:
                lines.append(f"{prefix}- {json.dumps(item)}")
        return "\n".join(lines) + "\n"
    return f"{prefix}{json.dumps(value)}\n"
