"""Model definition API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, model_definitions
from quant_platform.models.registry import ModelRegistry
from quant_platform.models.schemas import ModelDefinition, ModelType

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelCreateRequest(BaseModel):
    """Request payload for registering a model definition."""

    definition: ModelDefinition
    overwrite: bool = True


class ModelResponse(BaseModel):
    """Model definition returned by the API."""

    id: int
    name: str
    version: str
    model_type: ModelType
    parameters: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class ModelListResponse(BaseModel):
    """Collection response for registered model definitions."""

    models: list[ModelResponse]


def _catalog() -> MetadataCatalog:
    settings = get_settings()
    return MetadataCatalog(settings.catalog_db_path)


def _model_response(row: Any) -> ModelResponse:
    mapping = dict(row)
    return ModelResponse(
        id=mapping["id"],
        name=mapping["name"],
        version=mapping["version"],
        model_type=ModelType(mapping["model_type"]),
        parameters=mapping.get("parameters") or {},
        metadata=mapping.get("metadata") or {},
        created_at=mapping.get("created_at"),
    )


def _get_model_row(catalog: MetadataCatalog, model_id: int) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(
                select(model_definitions).where(model_definitions.c.id == model_id)
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"model definition not found: {model_id}",
        )
    return dict(row)


@router.get("", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """List registered model definitions."""

    catalog = _catalog()
    catalog.create_all()
    rows = catalog.list_rows("model_definitions")
    return ModelListResponse(models=[_model_response(row) for row in rows])


@router.post("", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
def create_model(request: ModelCreateRequest) -> ModelResponse:
    """Register a reusable model definition."""

    registry = ModelRegistry(get_settings().catalog_db_path)
    model_id = registry.register(request.definition, overwrite=request.overwrite)
    row = _get_model_row(registry.catalog, model_id)
    return _model_response(row)


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: int) -> ModelResponse:
    """Return a registered model definition by id."""

    catalog = _catalog()
    catalog.create_all()
    return _model_response(_get_model_row(catalog, model_id))
