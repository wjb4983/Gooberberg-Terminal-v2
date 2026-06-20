"""Ingestion manifest file and catalog writers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from quant_platform.common.paths import ensure_parent_directory
from quant_platform.data.ingestion.planner import IngestionPartition
from quant_platform.data.storage.catalog import MetadataCatalog

DEFAULT_MANIFEST_ROOT = Path("data/catalog/ingestion_manifests")


def file_fingerprint(path: Path) -> dict[str, Any]:
    """Return a cheap artifact fingerprint using size and a small content hash."""

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {"size_bytes": stat.st_size, "sha256_head": digest.hexdigest()}


@dataclass(frozen=True)
class IngestionManifest:
    """Metadata describing one completed ingestion partition."""

    manifest_id: str
    provider: str
    dataset: str
    symbol: str
    data_type: str
    date: str
    status: str
    artifact_uri: str
    row_count: int
    started_at: str
    completed_at: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest dictionary."""

        return asdict(self)


class ManifestWriter:
    """Write manifest JSON files and catalog rows."""

    def __init__(
        self,
        *,
        catalog: MetadataCatalog,
        root: str | Path = DEFAULT_MANIFEST_ROOT,
    ) -> None:
        self.catalog = catalog
        self.catalog.create_all()
        self.root = Path(root)

    def write_success(
        self,
        partition: IngestionPartition,
        *,
        artifact_path: Path,
        row_count: int,
        started_at: datetime,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[IngestionManifest, Path, int]:
        """Persist a successful ingestion manifest to disk and the catalog."""

        completed_at = datetime.now(UTC)
        manifest_id = uuid4().hex
        artifact_uri = artifact_path.as_posix()
        manifest = IngestionManifest(
            manifest_id=manifest_id,
            provider=partition.provider,
            dataset=partition.dataset,
            symbol=partition.symbol,
            data_type=partition.data_type,
            date=partition.date.isoformat(),
            status="succeeded",
            artifact_uri=artifact_uri,
            row_count=row_count,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            metadata={
                "asset_class": partition.asset_class,
                "resolution": partition.resolution,
                "fingerprint": file_fingerprint(artifact_path),
                **(metadata or {}),
            },
        )
        manifest_path = ensure_parent_directory(
            self.root
            / f"provider={partition.provider}"
            / f"dataset={partition.dataset}"
            / f"date={partition.date.isoformat()}"
            / f"{manifest_id}.json"
        )
        manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
        )
        row_id = self.catalog.insert_row(
            "ingestion_manifests",
            {
                "provider": partition.provider,
                "dataset": partition.dataset,
                "status": "succeeded",
                "source_uri": None,
                "artifact_uri": artifact_uri,
                "row_count": row_count,
                "started_at": started_at,
                "completed_at": completed_at,
                "metadata": {
                    **manifest.metadata,
                    "manifest_uri": manifest_path.as_posix(),
                },
            },
        )
        return manifest, manifest_path, row_id


__all__ = [
    "DEFAULT_MANIFEST_ROOT",
    "IngestionManifest",
    "ManifestWriter",
    "file_fingerprint",
]
