"""Tests for sanitized portfolio API routes."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from apps.api.main import app
from apps.api.routes import portfolio as portfolio_routes


class FakePortfolioService:
    def __init__(self, summary: dict[str, Any]) -> None:
        self._summary = summary

    def summary(self, **_kwargs: Any) -> dict[str, Any]:
        return self._summary


def _settings(
    client_id: str | None = "client-id", secret: str | None = "secret"
) -> Any:
    return SimpleNamespace(schwab_client_id=client_id, schwab_client_secret=secret)


def _base_summary(**overrides: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "totals": {
            "total_value": 125.0,
            "securities_value": 100.0,
            "cash_value": 25.0,
            "cost_basis": 90.0,
            "unrealized_pnl": 10.0,
        },
        "allocation_by_symbol": [
            {"symbol": "AAA", "market_value": 100.0, "weight": 0.8},
            {"symbol": "CASH", "market_value": 25.0, "weight": 0.2},
        ],
        "allocation_by_asset_type": [
            {"asset_type": "EQUITY", "market_value": 100.0, "weight": 0.8},
            {"asset_type": "CASH", "market_value": 25.0, "weight": 0.2},
        ],
        "holdings": [
            {
                "account_hash": "hash-secret",
                "account_number": "123456789",
                "masked_account_label": "Account ****6789",
                "raw_tokens": {"access_token": "secret-token"},
                "schwab_authorization_payload": {"code": "secret-code"},
                "symbol": "AAA",
                "asset_type": "EQUITY",
                "quantity": 1.0,
                "market_value": 100.0,
                "current_price": 100.0,
                "cost_basis": 90.0,
                "average_price": 90.0,
                "unrealized_pnl": 10.0,
                "unrealized_pnl_percent": 0.1111,
                "is_cash": False,
            }
        ],
        "lookback_metrics": {
            "MAX": {
                "lookback": "MAX",
                "weights": {"AAA": 0.8, "CASH": 0.2},
                "total_return": 0.1,
                "annualized_volatility": 0.2,
                "sharpe_ratio": 0.3,
                "beta": 1.0,
                "max_drawdown": -0.05,
            }
        },
        "warnings": [],
        "metadata": {
            "provider": "FakePortfolioService",
            "benchmark_symbol": "SPY",
            "refreshed_at": "2026-07-02T00:00:00+00:00",
            "holdings_refreshed_at": "2026-07-02T00:00:00+00:00",
            "prices_refreshed_at": "2026-07-02T00:00:00+00:00",
            "benchmark_refreshed_at": "2026-07-02T00:00:00+00:00",
            "account_hashes": ["hash-secret"],
            "lookbacks": ["MAX"],
        },
    }
    summary.update(overrides)
    return summary


def _asgi_get(path: str) -> tuple[int, dict[str, Any]]:
    async def _request() -> tuple[int, dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path.split("?", 1)[0],
            "raw_path": path.encode(),
            "query_string": path.split("?", 1)[1].encode() if "?" in path else b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            messages.append(message)

        await app(scope, receive, send)
        response_start = next(
            message for message in messages if message["type"] == "http.response.start"
        )
        response_body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        return int(response_start["status"]), json.loads(response_body or b"{}")

    return asyncio.run(_request())


def test_portfolio_holdings_success_sanitizes_secret_fields(monkeypatch) -> None:
    monkeypatch.setattr(portfolio_routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        portfolio_routes,
        "PortfolioService",
        lambda: FakePortfolioService(_base_summary()),
    )

    status, payload = _asgi_get("/api/v1/portfolio/holdings")

    assert status == 200
    body = json.dumps(payload)
    assert payload["totals"]["total_value"] == 125.0
    assert payload["holdings"] == [
        {
            "masked_account_label": "Account ****6789",
            "symbol": "AAA",
            "asset_type": "EQUITY",
            "quantity": 1.0,
            "market_value": 100.0,
            "current_price": 100.0,
            "cost_basis": 90.0,
            "average_price": 90.0,
            "unrealized_pnl": 10.0,
            "unrealized_pnl_percent": 0.1111,
            "is_cash": False,
        }
    ]
    assert "secret-token" not in body
    assert "secret-code" not in body
    assert "123456789" not in body
    assert "hash-secret" not in body


def test_portfolio_route_missing_schwab_configuration(monkeypatch) -> None:
    monkeypatch.setattr(
        portfolio_routes, "get_settings", lambda: _settings(client_id=None)
    )

    status, payload = _asgi_get("/api/v1/portfolio/metrics")

    assert status == 503
    assert payload == {"detail": "Schwab configuration is missing."}


def test_portfolio_holdings_empty_response(monkeypatch) -> None:
    monkeypatch.setattr(portfolio_routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        portfolio_routes,
        "PortfolioService",
        lambda: FakePortfolioService(
            _base_summary(
                totals={
                    "total_value": 0.0,
                    "securities_value": 0.0,
                    "cash_value": 0.0,
                    "cost_basis": 0.0,
                    "unrealized_pnl": 0.0,
                },
                allocation_by_symbol=[],
                allocation_by_asset_type=[],
                holdings=[],
            )
        ),
    )

    status, payload = _asgi_get("/api/v1/portfolio/holdings")

    assert status == 200
    assert payload["totals"]["total_value"] == 0.0
    assert payload["holdings"] == []


def test_portfolio_metrics_include_sanitized_service_warnings(monkeypatch) -> None:
    monkeypatch.setattr(portfolio_routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        portfolio_routes,
        "PortfolioService",
        lambda: FakePortfolioService(
            _base_summary(
                warnings=[
                    {
                        "code": "price_history_failed",
                        "message": "Unable to fetch price history for AAA.",
                        "symbol": "AAA",
                        "account_hash": "hash-secret",
                    }
                ]
            )
        ),
    )

    status, payload = _asgi_get("/api/v1/portfolio/metrics")

    assert status == 200
    assert payload["warnings"] == [
        {
            "code": "price_history_failed",
            "message": "Unable to fetch price history for AAA.",
            "symbol": "AAA",
        }
    ]
    assert "hash-secret" not in json.dumps(payload)


class FakePortfolioOptimizationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run_selected_strategies(self, **kwargs: Any) -> list[Any]:
        from quant_platform.portfolio.optimization import (
            OptimizedPortfolioResult,
            PortfolioOptimizationStrategy,
        )

        self.calls.append(kwargs)
        return [
            OptimizedPortfolioResult(
                strategy_id=PortfolioOptimizationStrategy.BUY_AND_HOLD,
                strategy_name="Buy-and-hold",
                target_weights={"AAA": 0.8, "CASH": 0.2},
                expected_return=0.1,
                volatility=0.2,
                sharpe=0.3,
                max_drawdown=-0.05,
                turnover=0.0,
                leverage=1.0,
                warnings=["Unable to fetch account hash-secret data."],
                constraints={
                    "benchmark_symbol": "QQQ",
                    "lookback": "1Y",
                    "account_hashes": ["hash-secret"],
                },
                is_placeholder=False,
            )
        ]


def test_portfolio_optimization_runs_selected_strategies_and_sanitizes(
    monkeypatch,
) -> None:
    fake_service = FakePortfolioOptimizationService()
    monkeypatch.setattr(portfolio_routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        portfolio_routes, "PortfolioOptimizationService", lambda: fake_service
    )

    status, payload = _asgi_get(
        "/api/v1/portfolio/optimization?strategy_ids=buy_and_hold"
        "&benchmark_symbol=qqq&lookback=1Y&account_hashes=hash-secret"
    )

    assert status == 200
    assert fake_service.calls == [
        {
            "account_hashes": ["hash-secret"],
            "benchmark_symbol": "QQQ",
            "lookback": "1Y",
            "strategy_ids": ["buy_and_hold"],
        }
    ]
    assert payload["metadata"] == {
        "benchmark_symbol": "QQQ",
        "lookback": "1Y",
        "selected_strategies": ["buy_and_hold"],
    }
    assert payload["results"][0]["strategy_id"] == "buy_and_hold"
    assert payload["results"][0]["target_weights"] == {"AAA": 0.8, "CASH": 0.2}
    assert "account_hashes" not in payload["results"][0]["constraints"]
    assert "hash-secret" not in json.dumps(payload["results"][0]["constraints"])
    assert payload["warnings"] == [
        {
            "code": "portfolio_optimization_warning",
            "message": "Unable to fetch account hash-secret data.",
            "symbol": None,
        }
    ]


def test_portfolio_optimization_rejects_unsupported_lookback(monkeypatch) -> None:
    class FailingPortfolioOptimizationService:
        def run_selected_strategies(self, **_kwargs: Any) -> list[Any]:
            raise ValueError("Unsupported lookback: BAD")

    monkeypatch.setattr(portfolio_routes, "get_settings", lambda: _settings())
    monkeypatch.setattr(
        portfolio_routes,
        "PortfolioOptimizationService",
        lambda: FailingPortfolioOptimizationService(),
    )

    status, payload = _asgi_get("/api/v1/portfolio/optimization?lookback=BAD")

    assert status == 422
    assert payload == {"detail": "Unsupported lookback: BAD"}
