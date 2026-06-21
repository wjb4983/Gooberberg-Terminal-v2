"""Monitoring summary service for model operations readiness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog


@dataclass(frozen=True)
class MonitoringStatus:
    """Feature status label shown by the monitoring API and UI."""

    status: str
    label: str
    detail: str


@dataclass(frozen=True)
class MonitoringSummary:
    """Current model-operations monitoring snapshot."""

    generated_at: datetime
    active_models: list[dict[str, Any]] = field(default_factory=list)
    recent_predictions: list[dict[str, Any]] = field(default_factory=list)
    data_freshness: list[dict[str, Any]] = field(default_factory=list)
    drift_checks: list[dict[str, Any]] = field(default_factory=list)
    latency: dict[str, Any] = field(default_factory=dict)
    trading_status: dict[str, MonitoringStatus] = field(default_factory=dict)

    def jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable monitoring snapshot."""

        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    """Convert dataclass payload values into JSON-safe primitives."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


class MonitoringService:
    """Build model monitoring summaries from the metadata catalog.

    Deployment, online prediction serving, drift jobs, latency collection, and live
    trading integrations are intentionally represented as not implemented until
    those platform capabilities are added.
    """

    def __init__(self, catalog: MetadataCatalog | None = None) -> None:
        self.catalog = catalog or MetadataCatalog(get_settings().catalog_db_path)

    def summary(self) -> MonitoringSummary:
        """Return a point-in-time monitoring summary for the UI and API."""

        self.catalog.create_all()
        models = [dict(row) for row in self.catalog.list_rows("model_definitions")]
        experiments = [dict(row) for row in self.catalog.list_rows("experiments")]
        manifests = [dict(row) for row in self.catalog.list_rows("ingestion_manifests")]
        coverage = [dict(row) for row in self.catalog.list_rows("coverage")]
        jobs = [dict(row) for row in self.catalog.list_rows("jobs")]

        return MonitoringSummary(
            generated_at=datetime.now(UTC),
            active_models=self._active_models(models, experiments),
            recent_predictions=self._recent_predictions(),
            data_freshness=self._data_freshness(manifests, coverage),
            drift_checks=self._drift_checks(),
            latency=self._latency(jobs),
            trading_status=self._trading_status(),
        )

    def _active_models(
        self, models: list[dict[str, Any]], experiments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        experiments_by_model: dict[int, list[dict[str, Any]]] = {}
        for experiment in experiments:
            model_id = experiment.get("model_id")
            if model_id is not None:
                experiments_by_model.setdefault(int(model_id), []).append(experiment)

        active_models: list[dict[str, Any]] = []
        for model in models:
            model_experiments = experiments_by_model.get(int(model["id"]), [])
            latest_experiment = max(
                model_experiments,
                key=lambda row: row.get("created_at") or datetime.min,
                default=None,
            )
            metadata = model.get("metadata") or {}
            active_models.append(
                {
                    "id": model["id"],
                    "name": model["name"],
                    "version": model.get("version"),
                    "model_type": model.get("model_type"),
                    "artifact_uri": model.get("artifact_uri"),
                    "deployment_status": metadata.get(
                        "deployment_status", "not_implemented"
                    ),
                    "serving_status": "not_implemented",
                    "latest_experiment_status": (latest_experiment or {}).get("status"),
                    "created_at": model.get("created_at"),
                }
            )
        return active_models

    def _recent_predictions(self) -> list[dict[str, Any]]:
        return [
            {
                "status": "not_implemented",
                "detail": "Online prediction logging is not implemented yet.",
            }
        ]

    def _data_freshness(
        self, manifests: list[dict[str, Any]], coverage: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        freshness: dict[str, dict[str, Any]] = {}
        for row in coverage:
            dataset = str(row.get("dataset") or "unknown")
            current = freshness.setdefault(
                dataset,
                {
                    "dataset": dataset,
                    "latest_data_ts": None,
                    "latest_ingestion_completed_at": None,
                    "row_count": 0,
                    "status": "available",
                },
            )
            end_ts = row.get("end_ts")
            if end_ts and (
                current["latest_data_ts"] is None or end_ts > current["latest_data_ts"]
            ):
                current["latest_data_ts"] = end_ts
            current["row_count"] += int(row.get("row_count") or 0)

        for manifest in manifests:
            dataset = str(manifest.get("dataset") or "unknown")
            current = freshness.setdefault(
                dataset,
                {
                    "dataset": dataset,
                    "latest_data_ts": None,
                    "latest_ingestion_completed_at": None,
                    "row_count": 0,
                    "status": manifest.get("status") or "unknown",
                },
            )
            completed_at = manifest.get("completed_at")
            if completed_at and (
                current["latest_ingestion_completed_at"] is None
                or completed_at > current["latest_ingestion_completed_at"]
            ):
                current["latest_ingestion_completed_at"] = completed_at
            current["status"] = manifest.get("status") or current["status"]

        return sorted(freshness.values(), key=lambda row: row["dataset"])

    def _drift_checks(self) -> list[dict[str, Any]]:
        return [
            {
                "status": "not_implemented",
                "detail": (
                    "Drift detection jobs and thresholds are not implemented yet."
                ),
            }
        ]

    def _latency(self, jobs: list[dict[str, Any]]) -> dict[str, Any]:
        running = sum(1 for job in jobs if job.get("status") in {"queued", "running"})
        completed = sum(
            1 for job in jobs if job.get("status") in {"completed", "succeeded"}
        )
        failed = sum(1 for job in jobs if job.get("status") == "failed")
        return {
            "queued_or_running_jobs": running,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "prediction_latency_status": "not_implemented",
            "detail": "Online serving latency collection is not implemented yet.",
        }

    def _trading_status(self) -> dict[str, MonitoringStatus]:
        return {
            "paper_trading": MonitoringStatus(
                status="not_implemented",
                label="Paper trading not implemented",
                detail=(
                    "Paper-trading execution and portfolio monitoring are placeholders."
                ),
            ),
            "live_trading": MonitoringStatus(
                status="not_implemented",
                label="Live trading not implemented",
                detail=(
                    "Live deployment and live trading are explicitly not implemented."
                ),
            ),
        }
