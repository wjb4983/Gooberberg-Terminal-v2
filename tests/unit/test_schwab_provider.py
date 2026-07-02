from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from quant_platform.config.settings import Settings
from quant_platform.data.providers.schwab import SchwabProvider, mask_account_label


class MockResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> MockResponse:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def make_settings(tmp_path: Path) -> Settings:
    token_path = tmp_path / "tokens.json"
    token_path.write_text(
        json.dumps({"access_token": "old-access", "refresh_token": "refresh-token"})
    )
    return Settings(
        schwab_client_id="client-id",
        schwab_client_secret="client-secret",
        schwab_token_path=token_path,
        schwab_api_timeout_seconds=4.25,
    )


def test_account_holdings_payload_normalization(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def transport(request, timeout: float):  # type: ignore[no-untyped-def]
        assert timeout == 4.25
        assert request.headers["Authorization"] == "Bearer old-access"
        return MockResponse(
            {
                "securitiesAccount": {
                    "accountNumber": "123456789",
                    "positions": [
                        {
                            "longQuantity": 3,
                            "marketValue": 450.0,
                            "currentPrice": 150.0,
                            "costBasis": 300.0,
                            "averagePrice": 100.0,
                            "currentDayProfitLoss": 150.0,
                            "currentDayProfitLossPercentage": 50.0,
                            "instrument": {"symbol": "MSFT", "assetType": "EQUITY"},
                        }
                    ],
                }
            }
        )

    provider = SchwabProvider(settings=settings, transport=transport)

    rows = provider.account_holdings("hash-abc").to_dicts()

    assert rows == [
        {
            "account_hash": "hash-abc",
            "masked_account_label": "Account ****6789",
            "symbol": "MSFT",
            "asset_type": "EQUITY",
            "quantity": 3.0,
            "market_value": 450.0,
            "current_price": 150.0,
            "cost_basis": 300.0,
            "average_price": 100.0,
            "unrealized_pnl": 150.0,
            "unrealized_pnl_percent": 50.0,
        }
    ]


def test_account_holdings_missing_fields_are_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def transport(_request, timeout: float):  # type: ignore[no-untyped-def]
        assert timeout == 4.25
        return MockResponse(
            {
                "securitiesAccount": {
                    "positions": [{"instrument": {"symbol": "CASH_EQUIVALENT"}}]
                }
            }
        )

    provider = SchwabProvider(settings=settings, transport=transport)

    row = provider.account_holdings("hash-missing").to_dicts()[0]

    assert row["account_hash"] == "hash-missing"
    assert row["masked_account_label"] == "Account ****"
    assert row["symbol"] == "CASH_EQUIVALENT"
    assert row["asset_type"] is None
    assert row["quantity"] is None
    assert row["market_value"] is None
    assert row["current_price"] is None
    assert row["cost_basis"] is None
    assert row["average_price"] is None
    assert row["unrealized_pnl"] is None
    assert row["unrealized_pnl_percent"] is None


def test_token_refresh_retries_after_unauthorized(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    calls: list[str] = []

    def transport(request, timeout: float):  # type: ignore[no-untyped-def]
        assert timeout == 4.25
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, 401, "unauthorized", hdrs=None, fp=None)
        if request.full_url.endswith("/v1/oauth/token"):
            assert request.headers["Authorization"].startswith("Basic ")
            body = parse_qs(request.data.decode())
            assert body == {
                "grant_type": ["refresh_token"],
                "refresh_token": ["refresh-token"],
            }
            return MockResponse(
                {"access_token": "new-access", "refresh_token": "new-refresh"}
            )
        assert request.headers["Authorization"] == "Bearer new-access"
        return MockResponse({"AAPL": {"symbol": "AAPL"}})

    provider = SchwabProvider(settings=settings, transport=transport)

    assert provider.quotes("AAPL") == {"AAPL": {"symbol": "AAPL"}}
    assert len(calls) == 3
    assert (
        json.loads(settings.schwab_token_path.read_text())["access_token"]
        == "new-access"
    )


def test_http_timeout_configuration_and_marketdata_urls(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    captured: dict[str, object] = {}

    def transport(request, timeout: float):  # type: ignore[no-untyped-def]
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        return MockResponse({"candles": []})

    provider = SchwabProvider(
        settings=settings, transport=transport, timeout_seconds=9.5
    )

    provider.historical_prices("msft", period=1)

    parsed = urlparse(str(captured["url"]))
    assert captured["timeout"] == 9.5
    assert parsed.path == "/marketdata/v1/pricehistory"
    assert parse_qs(parsed.query) == {
        "symbol": ["MSFT"],
        "periodType": ["day"],
        "period": ["1"],
        "frequencyType": ["daily"],
        "frequency": ["1"],
    }


def test_account_number_lookup_and_masking(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    def transport(_request, timeout: float):  # type: ignore[no-untyped-def]
        assert timeout == 4.25
        return MockResponse(
            [
                {"accountNumber": "123456789", "hashValue": "hash-1"},
                {"accountNumber": "abc-42", "hashValue": "hash-2"},
            ]
        )

    provider = SchwabProvider(settings=settings, transport=transport)

    accounts = provider.account_numbers()

    assert accounts[0].account_hash == "hash-1"
    assert accounts[0].masked_account_label == "Account ****6789"
    assert accounts[1].masked_account_label == "Account ****42"
    assert mask_account_label(None) == "Account ****"
