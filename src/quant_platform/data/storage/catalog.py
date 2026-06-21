"""SQLite-backed metadata catalog for platform assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine

from quant_platform.common.paths import ensure_parent_directory

DEFAULT_METADATA_DB_PATH = Path("data/catalog/metadata.sqlite")

metadata = MetaData()


def timestamp_column(name: str, *, onupdate: bool = False) -> Column[Any]:
    """Return a standard catalog timestamp column."""

    kwargs: dict[str, Any] = {"server_default": func.now(), "nullable": False}
    if onupdate:
        kwargs["onupdate"] = func.now()
    return Column(name, DateTime(timezone=True), **kwargs)


coverage = Table(
    "coverage",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("dataset", String(255), nullable=False, index=True),
    Column("symbol", String(64), nullable=False, index=True),
    Column("start_ts", DateTime(timezone=True), nullable=True),
    Column("end_ts", DateTime(timezone=True), nullable=True),
    Column("row_count", Integer, nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
    timestamp_column("updated_at", onupdate=True),
)

ingestion_manifests = Table(
    "ingestion_manifests",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("provider", String(128), nullable=False, index=True),
    Column("dataset", String(255), nullable=False, index=True),
    Column("status", String(64), nullable=False, default="pending"),
    Column("source_uri", Text, nullable=True),
    Column("artifact_uri", Text, nullable=True),
    Column("row_count", Integer, nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

dataset_definitions = Table(
    "dataset_definitions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), unique=True, nullable=False),
    Column("version", String(64), nullable=False, default="1"),
    Column("description", Text, nullable=True),
    Column("schema", JSON, nullable=False, default=dict),
    Column("storage_uri", Text, nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

model_definitions = Table(
    "model_definitions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), unique=True, nullable=False),
    Column("version", String(64), nullable=False, default="1"),
    Column("model_type", String(128), nullable=True),
    Column("artifact_uri", Text, nullable=True),
    Column("parameters", JSON, nullable=False, default=dict),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

feature_sets = Table(
    "feature_sets",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), unique=True, nullable=False),
    Column("version", String(64), nullable=False, default="1"),
    Column("features", JSON, nullable=False, default=list),
    Column("dataset_id", ForeignKey("dataset_definitions.id"), nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

experiments = Table(
    "experiments",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(255), nullable=False, index=True),
    Column("status", String(64), nullable=False, default="created"),
    Column("model_id", ForeignKey("model_definitions.id"), nullable=True),
    Column("dataset_id", ForeignKey("dataset_definitions.id"), nullable=True),
    Column("feature_set_id", ForeignKey("feature_sets.id"), nullable=True),
    Column("parameters", JSON, nullable=False, default=dict),
    Column("metadata", JSON, nullable=False, default=dict),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    timestamp_column("created_at"),
)

experiment_metrics = Table(
    "experiment_metrics",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("experiment_id", ForeignKey("experiments.id"), nullable=False, index=True),
    Column("name", String(255), nullable=False, index=True),
    Column("value", Float, nullable=False),
    Column("step", Integer, nullable=True),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

backtests = Table(
    "backtests",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("experiment_id", ForeignKey("experiments.id"), nullable=True, index=True),
    Column("name", String(255), nullable=False),
    Column("status", String(64), nullable=False, default="created"),
    Column("start_ts", DateTime(timezone=True), nullable=True),
    Column("end_ts", DateTime(timezone=True), nullable=True),
    Column("results", JSON, nullable=False, default=dict),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("job_type", String(128), nullable=False, index=True),
    Column("status", String(64), nullable=False, default="queued", index=True),
    Column("payload", JSON, nullable=False, default=dict),
    Column("result", JSON, nullable=True),
    Column("error", Text, nullable=True),
    timestamp_column("created_at"),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
)

job_logs = Table(
    "job_logs",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("job_id", ForeignKey("jobs.id"), nullable=False, index=True),
    Column("level", String(32), nullable=False, default="info"),
    Column("message", Text, nullable=False),
    Column("metadata", JSON, nullable=False, default=dict),
    timestamp_column("created_at"),
)

TABLES = {
    table.name: table
    for table in (
        coverage,
        ingestion_manifests,
        dataset_definitions,
        model_definitions,
        feature_sets,
        experiments,
        experiment_metrics,
        backtests,
        jobs,
        job_logs,
    )
}


def catalog_path(path: str | Path | None = None) -> Path:
    """Return the resolved metadata catalog path, creating its parent directory."""

    return ensure_parent_directory(path or DEFAULT_METADATA_DB_PATH)


def create_catalog_engine(path: str | Path | None = None) -> Engine:
    """Create a SQLite engine for the metadata catalog."""

    db_path = catalog_path(path)
    return create_engine(f"sqlite:///{db_path}", future=True)


def init_metadata_db(path: str | Path | None = None) -> Path:
    """Create all metadata tables and return the resolved SQLite database path."""

    db_path = catalog_path(path)
    engine = create_catalog_engine(db_path)
    metadata.create_all(engine)
    return db_path


class MetadataCatalog:
    """Small helper around SQLAlchemy Core tables for registry metadata."""

    def __init__(
        self,
        path: str | Path | None = None,
        engine: Engine | None = None,
    ) -> None:
        self.path = catalog_path(path)
        self.engine = engine or create_catalog_engine(self.path)

    def create_all(self) -> None:
        """Create catalog tables if they do not exist."""

        metadata.create_all(self.engine)

    def insert_row(self, table_name: str, values: Mapping[str, Any]) -> int:
        """Insert a row into a catalog table and return the inserted primary key."""

        table = TABLES[table_name]
        with self.engine.begin() as connection:
            result = connection.execute(insert(table).values(**dict(values)))
            inserted_primary_key = result.inserted_primary_key
            if inserted_primary_key is None:
                raise RuntimeError(f"insert into {table_name} did not return a key")
            inserted_id = cast(int, inserted_primary_key[0])
        return int(inserted_id)

    def update_row(
        self,
        table_name: str,
        row_id: int,
        values: Mapping[str, Any],
    ) -> int:
        """Update a catalog row by primary key and return the number of rows changed."""

        table = TABLES[table_name]
        with self.engine.begin() as connection:
            result = connection.execute(
                update(table).where(table.c.id == row_id).values(**dict(values))
            )
        return int(result.rowcount or 0)

    def list_rows(self, table_name: str) -> Sequence[Mapping[str, Any]]:
        """Return all rows from a catalog table as mappings."""

        table = TABLES[table_name]
        with self.engine.connect() as connection:
            rows = connection.execute(select(table)).mappings().all()
        return [dict(row) for row in rows]
