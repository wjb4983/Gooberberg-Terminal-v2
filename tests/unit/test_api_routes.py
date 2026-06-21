"""Tests for FastAPI route registration."""

from __future__ import annotations

from types import SimpleNamespace

from apps.api.main import app

from quant_platform.data.storage.catalog import MetadataCatalog


def test_api_v1_routes_are_registered() -> None:
    """Versioned API routes should be present in the OpenAPI schema."""

    paths = app.openapi()["paths"]
    expected_paths = {
        "/api/v1/health",
        "/api/v1/backtests",
        "/api/v1/backtests/{backtest_id}",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/datasets/{dataset_id}/coverage",
        "/api/v1/datasets/{dataset_id}/ingest",
        "/api/v1/ingestion/plan",
        "/api/v1/models",
        "/api/v1/models/{model_id}",
        "/api/v1/models/feature-sets",
        "/api/v1/models/feature-sets/{feature_set_id}",
        "/api/v1/experiments",
        "/api/v1/experiments/{experiment_id}",
        "/api/v1/ingestion/manifests",
        "/api/v1/jobs",
        "/api/v1/jobs/board",
        "/api/v1/jobs/{job_id}",
        "/api/v1/jobs/{job_id}/logs",
        "/api/v1/monitoring",
    }
    assert expected_paths <= set(paths)


def test_api_v1_routes_have_expected_methods() -> None:
    """Versioned API routes should expose the requested HTTP methods."""

    paths = app.openapi()["paths"]
    assert "get" in paths["/api/v1/health"]
    assert {"get", "post"} <= set(paths["/api/v1/backtests"])
    assert "get" in paths["/api/v1/backtests/{backtest_id}"]
    assert {"get", "post"} <= set(paths["/api/v1/datasets"])
    assert "get" in paths["/api/v1/datasets/{dataset_id}"]
    assert "post" in paths["/api/v1/datasets/{dataset_id}/coverage"]
    assert "post" in paths["/api/v1/datasets/{dataset_id}/ingest"]
    assert "post" in paths["/api/v1/ingestion/plan"]
    assert {"get", "post"} <= set(paths["/api/v1/models"])
    assert "get" in paths["/api/v1/models/{model_id}"]
    assert "get" in paths["/api/v1/models/feature-sets"]
    assert "get" in paths["/api/v1/models/feature-sets/{feature_set_id}"]
    assert {"get", "post"} <= set(paths["/api/v1/experiments"])
    assert "get" in paths["/api/v1/experiments/{experiment_id}"]
    assert "get" in paths["/api/v1/ingestion/manifests"]
    assert "get" in paths["/api/v1/jobs"]
    assert "get" in paths["/api/v1/jobs/board"]
    assert "get" in paths["/api/v1/jobs/{job_id}"]
    assert "get" in paths["/api/v1/jobs/{job_id}/logs"]
    assert "get" in paths["/api/v1/monitoring"]


def test_queue_experiment_uses_queue_helper(monkeypatch, tmp_path) -> None:
    """Queueing an experiment should delegate job creation to the queue helper."""

    from apps.api.routes import experiments as experiment_routes

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    dataset_id = catalog.insert_row(
        "dataset_definitions",
        {
            "name": "prices",
            "version": "v1",
            "schema": {},
            "metadata": {},
        },
    )
    model_id = catalog.insert_row(
        "model_definitions",
        {
            "name": "baseline",
            "version": "v1",
            "model_type": "lstm",
            "parameters": {},
            "metadata": {},
        },
    )
    feature_set_id = catalog.insert_row(
        "feature_sets",
        {
            "name": "basic-features",
            "version": "v1",
            "features": ["close", "volume"],
            "dataset_id": dataset_id,
            "metadata": {},
        },
    )

    monkeypatch.setattr(experiment_routes, "_catalog", lambda: catalog)
    queued_payloads = []

    def fake_enqueue_training_job(payload):
        queued_payloads.append(payload.copy())
        return SimpleNamespace(catalog_job_id=42, status="queued")

    monkeypatch.setattr(
        experiment_routes, "enqueue_training_job", fake_enqueue_training_job
    )

    response = experiment_routes.queue_experiment(
        experiment_routes.ExperimentQueueRequest.model_validate(
            {
                "name": " nightly training ",
                "dataset_id": dataset_id,
                "feature_set_id": feature_set_id,
                "model_id": model_id,
                "split": {
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-31",
                    "validation_start": "2024-02-01",
                    "validation_end": "2024-02-15",
                },
            }
        )
    )

    body = response.model_dump(mode="json")
    assert body["job_id"] == 42
    assert body["status"] == "queued"
    assert body["experiment_id"] == queued_payloads[0]["experiment_id"]
    assert queued_payloads == [body["payload"]]
    assert queued_payloads[0]["experiment_name"] == "nightly training"
    assert queued_payloads[0]["feature_set"] == ["close", "volume"]
