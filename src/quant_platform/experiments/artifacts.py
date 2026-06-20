"""Artifact writing utilities for experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from quant_platform.common.paths import ensure_directory
from quant_platform.experiments.schemas import ArtifactManifest


def experiment_artifact_dir(
    root: str | Path, experiment_name: str, experiment_id: int
) -> Path:
    """Return a stable artifact directory for an experiment."""

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in experiment_name
    )
    return ensure_directory(Path(root) / f"{experiment_id:06d}-{safe_name}")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_training_artifacts(
    *,
    artifact_dir: str | Path,
    experiment_id: int,
    experiment_name: str,
    model: nn.Module,
    config: dict[str, Any],
    metrics: dict[str, float],
    history: list[dict[str, float | int]],
    metadata: dict[str, Any] | None = None,
) -> ArtifactManifest:
    """Write model, config, metrics, and manifest artifacts."""

    root = ensure_directory(artifact_dir)
    model_path = root / "model.pt"
    config_path = root / "config.json"
    metrics_path = root / "metrics.json"
    history_path = root / "history.json"
    manifest_path = root / "manifest.json"

    torch.save(model.state_dict(), model_path)
    write_json(config_path, config)
    write_json(metrics_path, metrics)
    write_json(history_path, {"epochs": history})

    manifest = ArtifactManifest(
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        artifact_dir=root,
        files={
            "model": str(model_path),
            "config": str(config_path),
            "metrics": str(metrics_path),
            "history": str(history_path),
            "manifest": str(manifest_path),
        },
        metrics=metrics,
        metadata=dict(metadata or {}),
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest
