"""Massive market data provider skeleton.

The implementation keeps provider endpoint wrappers small and isolated while the
shared request layer handles authentication, request-manifest logging, and
conversion to Polars DataFrames. Unit tests should subclass or fake the request
layer; they should not call the live Massive API.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from quant_platform.config.settings import Settings, get_settings
from quant_platform.data.providers.base import BaseMarketDataProvider, DateLike

JsonObject = dict[str, Any]
RequestParams = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class RequestManifestEntry:
    """Request metadata suitable for ingestion manifests.

    The API key is intentionally omitted so this structure can be persisted or
    logged safely by ingestion jobs.
    """

    provider: str
    endpoint: str
    params: Mapping[str, str | int | float | bool]
    requested_at: datetime


@dataclass
class MassiveMarketDataProvider(BaseMarketDataProvider):
    """Massive API provider skeleton returning Polars DataFrames.

    The provider reads credentials from :class:`Settings` only. Pass a settings
    instance in tests when needed; do not pass API keys directly.
    """

    settings: Settings = field(default_factory=get_settings)
    base_url: str = "https://api.massive.com"
    timeout_seconds: float = 30.0
    name: str = "massive"
    _request_manifest: list[RequestManifestEntry] = field(default_factory=list)

    @property
    def request_manifest(self) -> tuple[RequestManifestEntry, ...]:
        """Return logged request parameters for ingestion manifests."""

        return tuple(self._request_manifest)

    def equity_trades(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity trade prints for ``symbol`` between ``start`` and ``end``."""

        endpoint = self._equity_trades_endpoint(symbol)
        params = self._date_range_params(start=start, end=end)
        return self._get_dataframe(endpoint, params=params)

    def equity_quotes(
        self, symbol: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return equity bid/ask quote updates for ``symbol``."""

        endpoint = self._equity_quotes_endpoint(symbol)
        params = self._date_range_params(start=start, end=end)
        return self._get_dataframe(endpoint, params=params)

    def aggregate_bars(
        self,
        symbol: str,
        *,
        start: DateLike,
        end: DateLike | None = None,
        resolution: str = "1min",
    ) -> pl.DataFrame:
        """Return OHLCV aggregate bars for ``symbol`` at ``resolution``."""

        multiplier, timespan = self._parse_resolution(resolution)
        endpoint = self._aggregate_bars_endpoint(
            symbol, multiplier, timespan, start, end
        )
        return self._get_dataframe(endpoint, params={"adjusted": True, "sort": "asc"})

    def option_chain(
        self, underlying: str, *, as_of: DateLike | None = None
    ) -> pl.DataFrame:
        """Return the option chain for ``underlying`` as of ``as_of``."""

        endpoint = self._option_chain_endpoint(underlying)
        params: RequestParams = {}
        if as_of is not None:
            params["as_of"] = self._format_date(as_of)
        return self._get_dataframe(endpoint, params=params)

    def option_contract_trades(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract trade prints for ``contract``."""

        endpoint = self._option_contract_trades_endpoint(contract)
        params = self._date_range_params(start=start, end=end)
        return self._get_dataframe(endpoint, params=params)

    def option_contract_quotes(
        self, contract: str, *, start: DateLike, end: DateLike | None = None
    ) -> pl.DataFrame:
        """Return option contract bid/ask quote updates for ``contract``."""

        endpoint = self._option_contract_quotes_endpoint(contract)
        params = self._date_range_params(start=start, end=end)
        return self._get_dataframe(endpoint, params=params)

    def snapshot(self, symbol: str) -> pl.DataFrame:
        """Return a latest market snapshot for ``symbol`` where supported."""

        endpoint = self._snapshot_endpoint(symbol)
        return self._get_dataframe(endpoint, params={})

    def _get_dataframe(self, endpoint: str, *, params: RequestParams) -> pl.DataFrame:
        self._log_request(endpoint, params)
        payload = self._request_json(endpoint, params)
        return self._payload_to_dataframe(payload)

    def _request_json(self, endpoint: str, params: RequestParams) -> JsonObject:
        api_key = self.settings.massive_api_key
        if not api_key:
            msg = "Massive API key is not configured; set MASSIVE_API_KEY."
            raise RuntimeError(msg)

        query = urlencode({**params, "apiKey": api_key})
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}?{query}"
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:  # noqa: S310
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            msg = f"Massive API request failed with HTTP {exc.code}: {endpoint}"
            raise RuntimeError(msg) from exc
        except URLError as exc:
            msg = f"Massive API request failed: {endpoint}"
            raise RuntimeError(msg) from exc

    def _log_request(self, endpoint: str, params: RequestParams) -> None:
        self._request_manifest.append(
            RequestManifestEntry(
                provider=self.name,
                endpoint=endpoint,
                params=dict(params),
                requested_at=datetime.now(UTC),
            )
        )

    def _payload_to_dataframe(self, payload: JsonObject) -> pl.DataFrame:
        rows = payload.get("results", payload.get("data", payload))
        if isinstance(rows, list):
            return pl.DataFrame(rows)
        if isinstance(rows, dict):
            return pl.DataFrame([rows])
        return pl.DataFrame()

    def _equity_trades_endpoint(self, symbol: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for equity trades.
        return f"v3/trades/{self._symbol(symbol)}"

    def _equity_quotes_endpoint(self, symbol: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for equity quotes.
        return f"v3/quotes/{self._symbol(symbol)}"

    def _aggregate_bars_endpoint(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        start: DateLike,
        end: DateLike | None,
    ) -> str:
        # TODO: Confirm exact Massive endpoint mapping for aggregate bars.
        from_date = self._format_date(start)
        to_date = self._format_date(end or start)
        return (
            f"v2/aggs/ticker/{self._symbol(symbol)}/range/"
            f"{multiplier}/{timespan}/{from_date}/{to_date}"
        )

    def _option_chain_endpoint(self, underlying: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for option chain snapshots.
        return f"v3/snapshot/options/{self._symbol(underlying)}"

    def _option_contract_trades_endpoint(self, contract: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for option contract trades.
        return f"v3/trades/{self._option_contract(contract)}"

    def _option_contract_quotes_endpoint(self, contract: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for option contract quotes.
        return f"v3/quotes/{self._option_contract(contract)}"

    def _snapshot_endpoint(self, symbol: str) -> str:
        # TODO: Confirm exact Massive endpoint mapping for latest equity snapshot.
        return f"v2/snapshot/locale/us/markets/stocks/tickers/{self._symbol(symbol)}"

    def _date_range_params(
        self, *, start: DateLike, end: DateLike | None = None
    ) -> RequestParams:
        params: RequestParams = {"timestamp.gte": self._format_date(start)}
        if end is not None:
            params["timestamp.lte"] = self._format_date(end)
        return params

    def _format_date(self, value: DateLike) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
        return value

    def _parse_resolution(self, resolution: str) -> tuple[int, str]:
        units = {"min": "minute", "minute": "minute", "hour": "hour", "day": "day"}
        for suffix, timespan in units.items():
            if resolution.endswith(suffix):
                raw_multiplier = resolution.removesuffix(suffix) or "1"
                return int(raw_multiplier), timespan
        msg = f"Unsupported Massive bar resolution: {resolution}"
        raise ValueError(msg)

    def _symbol(self, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            msg = "symbol cannot be empty"
            raise ValueError(msg)
        return symbol

    def _option_contract(self, value: str) -> str:
        contract = self._symbol(value)
        return contract if contract.startswith("O:") else f"O:{contract}"


__all__ = ["MassiveMarketDataProvider", "RequestManifestEntry"]
