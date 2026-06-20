"""Artifact writers for backtest runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict

from quant_platform.backtesting.schemas import BacktestConfig, BacktestResult
from quant_platform.common.paths import ensure_directory, ensure_parent_directory


class BacktestArtifactManifest(BaseModel):
    """Manifest describing persisted backtest artifacts."""

    model_config = ConfigDict(extra="forbid")

    artifact_dir: Path
    files: dict[str, str]
    metrics: dict[str, float | int]


def backtest_artifact_dir(root: str | Path, name: str) -> Path:
    """Return a stable artifact directory for a backtest name."""

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in name
    )
    return ensure_directory(Path(root) / safe_name)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write a JSON artifact and return the path."""

    target = ensure_parent_directory(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_backtest_artifacts(
    result: BacktestResult,
    *,
    artifact_dir: str | Path | None = None,
) -> BacktestArtifactManifest:
    """Save equity curve, trades, metrics, config snapshot, and manifest."""

    config: BacktestConfig = result.config
    root = ensure_directory(
        artifact_dir or backtest_artifact_dir(config.artifact_dir, config.name)
    )
    equity_path = root / "equity_curve.parquet"
    trades_path = root / "trades.parquet"
    metrics_path = root / "metrics.json"
    config_path = root / "config.json"
    manifest_path = root / "manifest.json"

    pd.DataFrame(result.equity_curve).to_parquet(equity_path, index=False)
    pd.DataFrame(result.trades).to_parquet(trades_path, index=False)
    write_json(metrics_path, dict(result.metrics))
    write_json(config_path, config.jsonable())

    manifest = BacktestArtifactManifest(
        artifact_dir=root,
        files={
            "equity_curve": str(equity_path),
            "trades": str(trades_path),
            "metrics": str(metrics_path),
            "config": str(config_path),
            "manifest": str(manifest_path),
        },
        metrics=result.metrics,
    )
    write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest
