"""Tests for FastAPI route registration."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from apps.api.main import app

from quant_platform.data.storage.catalog import MetadataCatalog


def _asgi_post_json(path: str, payload: dict[str, object]) -> tuple[int, bytes]:
    """POST JSON to the FastAPI ASGI app without requiring httpx/TestClient."""

    async def _request() -> tuple[int, bytes]:
        body = json.dumps(payload).encode()
        messages: list[dict[str, object]] = []
        received = False

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, object]:
            nonlocal received
            if received:
                return {"type": "http.disconnect"}
            received = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict[str, object]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        response_start = next(
            message for message in messages if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return int(response_start["status"]), response_body

    return asyncio.run(_request())


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


def test_post_experiment_creates_queued_job(monkeypatch, tmp_path) -> None:
    """POST /experiments should persist experiment and job rows without Redis."""

    from apps.api.routes import experiments as experiment_routes

    from quant_platform.jobs import queue as job_queue_module

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    dataset_id = catalog.insert_row(
        "dataset_definitions",
        {
            "name": "tiny-prices",
            "version": "v1",
            "schema": {"columns": ["close", "volume"]},
            "metadata": {"source": "unit-test"},
        },
    )
    feature_set_id = catalog.insert_row(
        "feature_sets",
        {
            "name": "tiny-features",
            "version": "v1",
            "features": ["close", "volume"],
            "dataset_id": dataset_id,
            "metadata": {},
        },
    )
    model_id = catalog.insert_row(
        "model_definitions",
        {
            "name": "tiny-lstm",
            "version": "v1",
            "model_type": "lstm",
            "parameters": {"hidden_size": 4},
            "metadata": {},
        },
    )

    monkeypatch.setattr(experiment_routes, "_catalog", lambda: catalog)
    monkeypatch.setattr(job_queue_module, "_catalog", lambda settings=None: catalog)

    class FakeQueue:
        name = "unit-test-training"

        def __init__(self) -> None:
            self.enqueued: list[dict[str, object]] = []

        def enqueue(self, task_path, catalog_job_id, payload, *, job_id):
            self.enqueued.append(
                {
                    "task_path": task_path,
                    "catalog_job_id": catalog_job_id,
                    "payload": dict(payload),
                    "job_id": job_id,
                }
            )
            return SimpleNamespace(id=job_id)

    fake_queue = FakeQueue()
    monkeypatch.setattr(
        job_queue_module,
        "job_queue",
        lambda name=job_queue_module.DEFAULT_QUEUE_NAME, **_: fake_queue,
    )

    status_code, response_body = _asgi_post_json(
        "/api/v1/experiments",
        {
            "name": "tiny api training",
            "dataset_id": dataset_id,
            "feature_set_id": feature_set_id,
            "model_id": model_id,
            "split": {
                "train_start": "2024-01-01",
                "train_end": "2024-01-02",
                "validation_start": "2024-01-03",
                "validation_end": "2024-01-04",
            },
            "training": {
                "epochs": 1,
                "batch_size": 2,
                "sequence_length": 2,
                "hidden_size": 4,
                "synthetic_rows_per_day": 1,
            },
        },
    )

    assert status_code == 201
    body = json.loads(response_body)
    assert body["status"] == "queued"
    assert body["payload"]["experiment_id"] == body["experiment_id"]

    experiment_rows = catalog.list_rows("experiments")
    assert len(experiment_rows) == 1
    assert experiment_rows[0]["id"] == body["experiment_id"]
    assert experiment_rows[0]["status"] == "queued"

    job_rows = catalog.list_rows("jobs")
    assert len(job_rows) == 1
    assert job_rows[0]["id"] == body["job_id"]
    assert job_rows[0]["status"] == "queued"
    assert job_rows[0]["payload"]["experiment_id"] == body["experiment_id"]

    assert len(fake_queue.enqueued) == 1
    queued_call = fake_queue.enqueued[0]
    assert queued_call["catalog_job_id"] == body["job_id"]
    assert queued_call["payload"]["experiment_id"] == body["experiment_id"]


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


def test_queue_experiment_rejects_future_kind_before_enqueue(monkeypatch, tmp_path):
    """Unsupported future experiment kinds should return validation errors."""

    import pytest
    from apps.api.routes import experiments as experiment_routes
    from fastapi import HTTPException

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
            "name": "markov-baseline",
            "version": "v1",
            "model_type": None,
            "parameters": {},
            "metadata": {"model_family": "markov"},
        },
    )

    monkeypatch.setattr(experiment_routes, "_catalog", lambda: catalog)

    def fail_enqueue_training_job(payload):
        raise AssertionError("unsupported experiments must not be enqueued")

    monkeypatch.setattr(
        experiment_routes, "enqueue_training_job", fail_enqueue_training_job
    )

    with pytest.raises(HTTPException) as exc_info:
        experiment_routes.queue_experiment(
            experiment_routes.ExperimentQueueRequest.model_validate(
                {
                    "name": "markov experiment",
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "experiment_kind": "markov_model",
                    "split": {
                        "train_start": "2024-01-01",
                        "train_end": "2024-01-31",
                        "validation_start": "2024-02-01",
                        "validation_end": "2024-02-15",
                    },
                }
            )
        )

    assert exc_info.value.status_code == 422
    assert "unsupported experiment kind for queueing" in str(exc_info.value.detail)
