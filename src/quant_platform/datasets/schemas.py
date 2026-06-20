"""Schemas for logical dataset definitions."""

from __future__ import annotations

from datetime import date, datetime
from importlib.util import find_spec
from typing import Any, get_type_hints

from quant_platform.common.enums import DataType, Provider

if find_spec("pydantic") is not None:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
else:

    class BaseModel:  # type: ignore[no-redef]
        """Small fallback used when Pydantic is not installed locally."""

        def __init__(self, **data: Any) -> None:
            annotations = get_type_hints(self.__class__)
            for name in annotations:
                value = data.get(name, getattr(self.__class__, name, None))
                setattr(self, name, value)

        def model_dump(self, **_: Any) -> dict[str, Any]:
            return dict(self.__dict__)

    ConfigDict = dict  # type: ignore[assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

    def field_validator(*_: str, **__: Any):  # type: ignore[no-redef]
        def decorator(func: Any) -> Any:
            return func

        return decorator

    def model_validator(*_: Any, **__: Any):  # type: ignore[no-redef]
        def decorator(func: Any) -> Any:
            return func

        return decorator


class DateRange(BaseModel):
    """Inclusive date boundaries for a dataset view."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(frozen=True)

    start: date | None = Field(default=None, description="Inclusive start date.")
    end: date | None = Field(default=None, description="Inclusive end date.")

    @model_validator(mode="after")
    def _validate_order(self) -> DateRange:
        if self.start is not None and self.end is not None and self.start > self.end:
            msg = "date range start must be on or before end"
            raise ValueError(msg)
        return self


class FeatureSetReference(BaseModel):
    """Reference to an optional feature set consumed by a logical dataset."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(frozen=True)

    name: str
    version: str | None = None
    id: int | None = None


class DatasetDefinition(BaseModel):
    """Logical view definition over raw or derived platform data."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(
            use_enum_values=False,
            arbitrary_types_allowed=True,
            extra="forbid",
        )

    name: str = Field(description="Unique logical dataset name.")
    version: str = Field(default="1", description="Dataset definition version.")
    asset_universe: list[str] = Field(
        default_factory=list,
        description="Symbols, universe names, or asset selectors included in the view.",
    )
    provider: Provider = Field(description="Provider backing the source data.")
    data_types: list[DataType] = Field(
        description="Market/reference data types included."
    )
    resolution: str | None = Field(
        default=None,
        description="Logical granularity, for example 1m, 1d, tick, or derived.",
    )
    date_range: DateRange = Field(default_factory=DateRange)
    filters: dict[str, Any] = Field(default_factory=dict)
    feature_set: FeatureSetReference | None = None
    description: str | None = None
    source: str = Field(
        default="logical_view",
        description="Human-readable source/view type. Does not imply materialization.",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("asset_universe", "data_types")
    @classmethod
    def _require_non_empty_list(cls, value: list[Any]) -> list[Any]:
        if not value:
            msg = "must contain at least one value"
            raise ValueError(msg)
        return value

    def to_catalog_schema(self) -> dict[str, Any]:
        """Return the JSON schema payload persisted in the metadata catalog."""

        return self.model_dump(mode="json", exclude={"description", "metadata"})

    def to_config_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation of the definition."""

        return self.model_dump(mode="json")


class DatasetMaterializationPlan(BaseModel):
    """Description of how a logical dataset can be read or optionally written."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(extra="forbid")

    dataset: DatasetDefinition
    logical: bool = True
    query: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
