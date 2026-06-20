"""Environment-backed application settings."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec
from os import getenv
from pathlib import Path
from typing import Any, get_type_hints

from quant_platform.common.enums import Provider
from quant_platform.common.paths import expand_path

if find_spec("pydantic") is not None:
    from pydantic import AnyUrl, Field, field_validator
else:
    AnyUrl = str

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        return default

    def field_validator(*_: str, **__: Any):  # type: ignore[no-redef]
        def decorator(func: Any) -> Any:
            return func

        return decorator

if find_spec("pydantic_settings") is not None:
    from pydantic_settings import BaseSettings, SettingsConfigDict
elif find_spec("pydantic") is not None:
    from pydantic import BaseSettings  # type: ignore[no-redef]

    SettingsConfigDict = dict  # type: ignore[assignment]
else:

    class BaseSettings:  # type: ignore[no-redef]
        """Small fallback used when Pydantic settings is not installed locally."""

        def __init__(self, **data: Any) -> None:
            annotations = get_type_hints(self.__class__)
            for field_name in annotations:
                default = getattr(self.__class__, field_name, None)
                env_name = f"QUANT_PLATFORM_{field_name}".upper()
                if field_name == "massive_api_key":
                    value = data.get(
                        field_name,
                        getenv("MASSIVE_API_KEY", getenv(env_name, default)),
                    )
                else:
                    value = data.get(field_name, getenv(env_name, default))
                if field_name in {"data_lake_root", "catalog_db_path"}:
                    value = expand_path(value)
                elif field_name == "default_provider" and not isinstance(
                    value, Provider
                ):
                    value = Provider(value)
                setattr(self, field_name, value)

    SettingsConfigDict = dict  # type: ignore[assignment]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and ``.env`` files."""

    if find_spec("pydantic") is not None:
        model_config = SettingsConfigDict(
            env_prefix="QUANT_PLATFORM_",
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

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
        validation_alias="MASSIVE_API_KEY",
        description="Massive API key.",
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance for process-wide reuse."""

    return Settings()
