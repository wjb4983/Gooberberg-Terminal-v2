"""Tests for training task job and experiment lifecycle updates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_platform.config.settings import Settings
from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.jobs import tasks


def _configure_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> MetadataCatalog:
    catalog_path = tmp_path / "metadata.sqlite"
    catalog = MetadataCatalog(catalog_path)
    catalog.create_all()
    settings = Settings(catalog_db_path=catalog_path, data_lake_root=tmp_path / "lake")
    monkeypatch.setattr(tasks, "get_settings", lambda: settings)
    return catalog


def _insert_training_rows(catalog: MetadataCatalog) -> tuple[int, int]:
    experiment_id = catalog.insert_row(
        "experiments",
        {
            "name": "queued training",
            "status": "queued",
            "parameters": {},
            "metadata": {"source": "test"},
        },
    )
    job_id = catalog.insert_row(
        "jobs",
        {
            "job_type": "training",
            "status": "queued",
            "payload": {"experiment_id": experiment_id},
        },
    )
    return experiment_id, job_id


def _row(catalog: MetadataCatalog, table_name: str, row_id: int) -> dict:
    rows = [row for row in catalog.list_rows(table_name) if row["id"] == row_id]
    assert len(rows) == 1
    return dict(rows[0])


def test_run_training_job_updates_job_and_experiment_on_success(monkeypatch, tmp_path):
    """Successful training should mark both lifecycle rows complete with artifacts."""

    catalog = _configure_catalog(monkeypatch, tmp_path)
    experiment_id, job_id = _insert_training_rows(catalog)
    calls = []

    def fake_run_training(config):
        calls.append(config)
        manifest = SimpleNamespace(files={"model": "artifacts/model.pt"})
        return SimpleNamespace(
            artifact_dir=tmp_path / "artifacts" / "run-1",
            manifest=manifest,
            model_dump=lambda mode="json": {
                "experiment_id": experiment_id,
                "artifact_dir": "artifacts/run-1",
                "manifest": {"files": {"model": "artifacts/model.pt"}},
            },
        )

    monkeypatch.setattr(tasks, "run_training", fake_run_training)

    result = tasks.run_training_job(
        job_id,
        {
            "experiment_id": experiment_id,
            "experiment_name": "queued training",
            "split": {
                "train_start": "2024-01-01",
                "train_end": "2024-01-31",
                "validation_start": "2024-02-01",
                "validation_end": "2024-02-15",
            },
            "training": {"epochs": 1, "batch_size": 2},
        },
    )

    assert calls[0].experiment_id == experiment_id
    assert calls[0].date_split.train_start.isoformat() == "2024-01-01"
    assert result["experiment_id"] == experiment_id
    job = _row(catalog, "jobs", job_id)
    experiment = _row(catalog, "experiments", experiment_id)
    assert job["status"] == "succeeded"
    assert job["result"] == result
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert experiment["status"] == "succeeded"
    assert experiment["started_at"] is not None
    assert experiment["completed_at"] is not None
    assert experiment["metadata"]["source"] == "test"
    assert experiment["metadata"]["artifacts"] == {
        "artifact_dir": str(tmp_path / "artifacts" / "run-1"),
        "model": "artifacts/model.pt",
    }


def test_run_training_job_updates_job_and_experiment_on_failure(monkeypatch, tmp_path):
    """Failed training should mark both lifecycle rows failed with concise errors."""

    catalog = _configure_catalog(monkeypatch, tmp_path)
    experiment_id, job_id = _insert_training_rows(catalog)

    def fake_run_training(config):
        raise RuntimeError("training exploded with a concise error")

    monkeypatch.setattr(tasks, "run_training", fake_run_training)

    with pytest.raises(RuntimeError, match="training exploded"):
        tasks.run_training_job(
            job_id,
            {
                "experiment_id": experiment_id,
                "experiment_name": "queued training",
                "date_split": {
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-31",
                    "validation_start": "2024-02-01",
                    "validation_end": "2024-02-15",
                },
            },
        )

    job = _row(catalog, "jobs", job_id)
    experiment = _row(catalog, "experiments", experiment_id)
    assert job["status"] == "failed"
    assert job["error"] == "training exploded with a concise error"
    assert job["started_at"] is not None
    assert job["completed_at"] is not None
    assert experiment["status"] == "failed"
    assert experiment["started_at"] is not None
    assert experiment["completed_at"] is not None
    assert experiment["metadata"]["source"] == "test"
    assert experiment["metadata"]["error"] == "training exploded with a concise error"
