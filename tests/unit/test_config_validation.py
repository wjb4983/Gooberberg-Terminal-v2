from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_platform.common.enums import Provider
from quant_platform.config.schemas import PlatformConfig, ProviderCredentials
from quant_platform.config.settings import Settings


def test_platform_config_expands_temp_directory_paths(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    catalog = tmp_path / "catalog" / "metadata.sqlite"

    config = PlatformConfig(data_lake_root=lake, catalog_db_path=catalog)

    assert config.data_lake_root == lake.resolve()
    assert config.catalog_db_path == catalog.resolve()


def test_settings_accepts_fake_provider_and_temp_paths(tmp_path: Path) -> None:
    settings = Settings(
        data_lake_root=tmp_path / "lake",
        catalog_db_path=tmp_path / "catalog" / "metadata.sqlite",
        default_provider=Provider.INTERNAL,
        massive_api_key=None,
    )

    assert settings.data_lake_root == (tmp_path / "lake").resolve()
    expected_catalog = (tmp_path / "catalog" / "metadata.sqlite").resolve()
    assert settings.catalog_db_path == expected_catalog
    assert settings.default_provider is Provider.INTERNAL


def test_invalid_provider_is_rejected() -> None:
    with pytest.raises((ValidationError, ValueError)):
        PlatformConfig(default_provider="not-a-provider")


def test_provider_credentials_allow_fake_provider_without_api_key() -> None:
    credentials = ProviderCredentials(provider=Provider.INTERNAL)

    assert credentials.provider is Provider.INTERNAL
    assert credentials.api_key is None
    assert credentials.base_url is None
