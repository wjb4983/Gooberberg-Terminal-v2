"""Pydantic schemas for configuration values."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, get_type_hints

from quant_platform.common.enums import Provider
from quant_platform.common.paths import expand_path

if find_spec("pydantic") is not None:
    from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator
else:
    AnyUrl = str

    class BaseModel:  # type: ignore[no-redef]
        """Small fallback used when Pydantic is not installed locally."""

        def __init__(self, **data: Any) -> None:
            annotations = get_type_hints(self.__class__)
            for name in annotations:
                if name in data:
                    value = data[name]
                else:
                    value = getattr(self.__class__, name, None)
                if name in {"data_lake_root", "catalog_db_path"}:
                    value = expand_path(value)
                elif name == "default_provider" and not isinstance(value, Provider):
                    value = Provider(value)
                setattr(self, name, value)

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

    def field_validator(*_: str, **__: Any):  # type: ignore[no-redef]
        def decorator(func: Any) -> Any:
            return func

        return decorator

    ConfigDict = dict  # type: ignore[assignment]


class PlatformConfig(BaseModel):
    """Core configuration shared by API, worker, and research processes."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(use_enum_values=False, arbitrary_types_allowed=True)

    data_lake_root: Path = Field(
        default=Path("./data/lake"),
        description="Root directory for data lake files.",
    )
    catalog_db_path: Path = Field(
        default=Path("./data/catalog/catalog.db"),
        description="SQLite catalog database path.",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL.",
    )
    massive_api_key: str | None = Field(
        default=None,
        description="Massive API key loaded from the environment.",
    )
    default_provider: Provider = Field(
        default=Provider.MASSIVE,
        description="Default market data provider.",
    )
    api_base_url: AnyUrl | str = Field(
        default="http://localhost:8000",
        description="Base URL for the platform API.",
    )

    @field_validator("data_lake_root", "catalog_db_path", mode="before")
    @classmethod
    def _expand_paths(cls, value: str | Path) -> Path:
        return expand_path(value)


class ProviderCredentials(BaseModel):
    """Optional provider-specific credential bundle."""

    if find_spec("pydantic") is not None:
        model_config = ConfigDict(use_enum_values=False)

    provider: Provider
    api_key: str | None = None
    base_url: AnyUrl | str | None = None
