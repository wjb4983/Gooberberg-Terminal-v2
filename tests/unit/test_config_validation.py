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


def test_schwab_settings_defaults_are_non_secret_and_serializable() -> None:
    settings = Settings(
        massive_api_key="massive-secret", schwab_client_secret="schwab-secret"
    )

    assert settings.schwab_client_id is None
    assert settings.schwab_client_secret == "schwab-secret"
    assert str(settings.schwab_redirect_uri) == "https://127.0.0.1:8182/callback"
    assert (
        settings.schwab_token_path
        == Path("./data/secrets/schwab_tokens.json").resolve()
    )
    assert settings.schwab_api_timeout_seconds == 30.0
    assert settings.schwab_default_benchmark == "SPY"
    assert settings.schwab_supported_lookback_windows == [
        "1d",
        "5d",
        "1m",
        "3m",
        "6m",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
    ]

    dumped = settings.model_dump()
    assert "massive_api_key" not in dumped
    assert "schwab_client_secret" not in dumped
    assert "massive-secret" not in repr(settings)
    assert "schwab-secret" not in repr(settings)


def test_schwab_settings_can_be_overridden_from_environment(
    monkeypatch, tmp_path: Path
) -> None:
    token_path = tmp_path / "schwab" / "tokens.json"
    monkeypatch.setenv("QUANT_PLATFORM_SCHWAB_CLIENT_ID", "placeholder-client-id")
    monkeypatch.setenv(
        "QUANT_PLATFORM_SCHWAB_CLIENT_SECRET", "placeholder-client-secret"
    )
    monkeypatch.setenv(
        "QUANT_PLATFORM_SCHWAB_REDIRECT_URI", "https://localhost:9443/callback"
    )
    monkeypatch.setenv("QUANT_PLATFORM_SCHWAB_TOKEN_PATH", str(token_path))
    monkeypatch.setenv("QUANT_PLATFORM_SCHWAB_API_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("QUANT_PLATFORM_SCHWAB_DEFAULT_BENCHMARK", "QQQ")
    monkeypatch.setenv(
        "QUANT_PLATFORM_SCHWAB_SUPPORTED_LOOKBACK_WINDOWS",
        '["1d", "30d", "1y"]',
    )

    settings = Settings()

    assert settings.schwab_client_id == "placeholder-client-id"
    assert settings.schwab_client_secret == "placeholder-client-secret"
    assert str(settings.schwab_redirect_uri) == "https://localhost:9443/callback"
    assert settings.schwab_token_path == token_path.resolve()
    assert settings.schwab_api_timeout_seconds == 12.5
    assert settings.schwab_default_benchmark == "QQQ"
    assert settings.schwab_supported_lookback_windows == ["1d", "30d", "1y"]
