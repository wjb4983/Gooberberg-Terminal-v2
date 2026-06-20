"""Model definition API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import (
    MetadataCatalog,
    feature_sets,
    model_definitions,
)
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


class FeatureSetResponse(BaseModel):
    """Feature set returned by the API for experiment configuration."""

    id: int
    name: str
    version: str
    features: list[str] = Field(default_factory=list)
    dataset_id: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class FeatureSetListResponse(BaseModel):
    """Collection response for registered feature sets."""

    feature_sets: list[FeatureSetResponse]


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


def _feature_set_response(row: Any) -> FeatureSetResponse:
    mapping = dict(row)
    return FeatureSetResponse(
        id=mapping["id"],
        name=mapping["name"],
        version=mapping["version"],
        features=list(mapping.get("features") or []),
        dataset_id=mapping.get("dataset_id"),
        metadata=mapping.get("metadata") or {},
        created_at=mapping.get("created_at"),
    )


def _get_feature_set_row(
    catalog: MetadataCatalog, feature_set_id: int
) -> dict[str, Any]:
    with catalog.engine.connect() as connection:
        row = (
            connection.execute(
                select(feature_sets).where(feature_sets.c.id == feature_set_id)
            )
            .mappings()
            .first()
        )
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"feature set not found: {feature_set_id}",
        )
    return dict(row)


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


@router.get("/feature-sets", response_model=FeatureSetListResponse)
def list_feature_sets() -> FeatureSetListResponse:
    """List registered feature sets for experiment configuration."""

    catalog = _catalog()
    catalog.create_all()
    rows = catalog.list_rows("feature_sets")
    return FeatureSetListResponse(
        feature_sets=[_feature_set_response(row) for row in rows]
    )


@router.get("/feature-sets/{feature_set_id}", response_model=FeatureSetResponse)
def get_feature_set(feature_set_id: int) -> FeatureSetResponse:
    """Return a registered feature set by id."""

    catalog = _catalog()
    catalog.create_all()
    return _feature_set_response(_get_feature_set_row(catalog, feature_set_id))


@router.get("/{model_id}", response_model=ModelResponse)
def get_model(model_id: int) -> ModelResponse:
    """Return a registered model definition by id."""

    catalog = _catalog()
    catalog.create_all()
    return _model_response(_get_model_row(catalog, model_id))
