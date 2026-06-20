from __future__ import annotations

import polars as pl

from quant_platform.config.settings import Settings
from quant_platform.data.providers.base import MarketDataProvider
from quant_platform.data.providers.massive import MassiveMarketDataProvider


class FakeMassiveProvider(MassiveMarketDataProvider):
    def _request_json(
        self, endpoint: str, params: dict[str, str | int | float | bool]
    ) -> dict[str, object]:
        return {
            "results": [
                {
                    "endpoint": endpoint,
                    "symbol": "AAPL",
                    "price": 100.0,
                    "size": 10,
                }
            ]
        }


def test_massive_provider_protocol_no_live_call(monkeypatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "fake-key")
    provider = FakeMassiveProvider(settings=Settings())

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "massive"


def test_massive_methods_return_dataframes_and_log_params(monkeypatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "fake-key")
    provider = FakeMassiveProvider(settings=Settings())

    trades = provider.equity_trades("aapl", start="2026-06-20", end="2026-06-21")
    bars = provider.aggregate_bars("msft", start="2026-06-20", resolution="5min")

    assert isinstance(trades, pl.DataFrame)
    assert isinstance(bars, pl.DataFrame)
    assert trades.item(0, "endpoint") == "v3/trades/AAPL"
    assert bars.item(0, "endpoint") == (
        "v2/aggs/ticker/MSFT/range/5/minute/2026-06-20/2026-06-20"
    )
    assert len(provider.request_manifest) == 2
    assert provider.request_manifest[0].params == {
        "timestamp.gte": "2026-06-20",
        "timestamp.lte": "2026-06-21",
    }
    assert "fake-key" not in str(provider.request_manifest)


def test_massive_api_key_is_read_from_settings_only(monkeypatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "env-key")
    settings = Settings()
    provider = FakeMassiveProvider(settings=settings)

    snapshot = provider.snapshot("nvda")

    assert settings.massive_api_key == "env-key"
    assert snapshot.item(0, "endpoint").endswith("/NVDA")
    assert "env-key" not in str(provider.request_manifest)
