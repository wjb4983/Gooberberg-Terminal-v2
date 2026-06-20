"""Tests for FastAPI route registration."""

from __future__ import annotations

from apps.api.main import app


def test_api_v1_routes_are_registered() -> None:
    """Versioned API routes should be present in the OpenAPI schema."""

    paths = app.openapi()["paths"]
    expected_paths = {
        "/api/v1/health",
        "/api/v1/datasets",
        "/api/v1/datasets/{dataset_id}",
        "/api/v1/datasets/{dataset_id}/coverage",
        "/api/v1/datasets/{dataset_id}/ingest",
        "/api/v1/ingestion/plan",
        "/api/v1/ingestion/manifests",
        "/api/v1/jobs",
        "/api/v1/jobs/{job_id}",
    }
    assert expected_paths <= set(paths)


def test_api_v1_routes_have_expected_methods() -> None:
    """Versioned API routes should expose the requested HTTP methods."""

    paths = app.openapi()["paths"]
    assert "get" in paths["/api/v1/health"]
    assert {"get", "post"} <= set(paths["/api/v1/datasets"])
    assert "get" in paths["/api/v1/datasets/{dataset_id}"]
    assert "post" in paths["/api/v1/datasets/{dataset_id}/coverage"]
    assert "post" in paths["/api/v1/datasets/{dataset_id}/ingest"]
    assert "post" in paths["/api/v1/ingestion/plan"]
    assert "get" in paths["/api/v1/ingestion/manifests"]
    assert "get" in paths["/api/v1/jobs"]
    assert "get" in paths["/api/v1/jobs/{job_id}"]
