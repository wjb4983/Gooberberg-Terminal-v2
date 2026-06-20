"""Schemas for experiment results and artifacts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ArtifactManifest(BaseModel):
    """Manifest describing files emitted by a training experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: int | None = None
    experiment_name: str
    artifact_dir: Path
    files: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentRunResult(BaseModel):
    """Result returned by the training runner."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: int
    experiment_name: str
    artifact_dir: Path
    device: str
    metrics: dict[str, float]
    history: list[dict[str, float | int]]
    manifest: ArtifactManifest
