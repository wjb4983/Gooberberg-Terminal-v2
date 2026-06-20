"""FastAPI entrypoint for the Gooberberg Terminal API."""

from __future__ import annotations

from fastapi import FastAPI

from quant_platform.config import get_settings

app = FastAPI(title="Gooberberg Terminal API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight health check response."""

    return {"status": "ok"}


@app.get("/settings")
def settings_summary() -> dict[str, str]:
    """Return non-secret runtime settings useful for local development."""

    settings = get_settings()
    return {
        "api_base_url": str(settings.api_base_url),
        "data_lake_root": str(settings.data_lake_root),
        "catalog_db_path": str(settings.catalog_db_path),
        "default_provider": str(settings.default_provider),
    }
