"""Schwab API provider adapter with injectable transport for tests.

The adapter intentionally keeps HTTP transport injectable so unit tests can pass
fake openers and never touch live Schwab endpoints.
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl

from quant_platform.cli.schwab_auth import SCHWAB_TOKEN_URL, SchwabAuthError
from quant_platform.config.settings import Settings, get_settings

JsonObject = dict[str, Any]
Transport = Callable[..., Any]

SCHWAB_API_BASE_URL = "https://api.schwabapi.com"


class SchwabProviderError(RuntimeError):
    """Raised when Schwab provider requests or normalization fail."""


@dataclass(frozen=True)
class SchwabAccount:
    """Normalized account-number lookup entry."""

    account_number: str | None
    account_hash: str
    masked_account_label: str


@dataclass
class SchwabProvider:
    """Provider adapter for Schwab account, quote, and price-history APIs."""

    settings: Settings = field(default_factory=get_settings)
    base_url: str = SCHWAB_API_BASE_URL
    token_url: str = SCHWAB_TOKEN_URL
    transport: Transport = urlopen
    timeout_seconds: float | None = None
    _tokens: JsonObject | None = field(default=None, init=False, repr=False)

    def account_numbers(self) -> list[SchwabAccount]:
        """Return Schwab account numbers with their encrypted account hashes."""

        payload = self._get_json("/trader/v1/accounts/accountNumbers")
        if not isinstance(payload, list):
            raise SchwabProviderError("Schwab account-number payload was not a list.")
        return [
            SchwabAccount(
                account_number=self._optional_str(row.get("accountNumber")),
                account_hash=self._required_str(row.get("hashValue"), "hashValue"),
                masked_account_label=mask_account_label(row.get("accountNumber")),
            )
            for row in payload
            if isinstance(row, Mapping)
        ]

    def account_holdings(self, account_hash: str) -> pl.DataFrame:
        """Return normalized holdings for an account hash."""

        payload = self._get_json(
            f"/trader/v1/accounts/{account_hash}", params={"fields": "positions"}
        )
        if not isinstance(payload, Mapping):
            raise SchwabProviderError("Schwab account payload was not a JSON object.")
        account = payload.get("securitiesAccount", payload)
        account_number = (
            account.get("accountNumber") if isinstance(account, Mapping) else None
        )
        positions = account.get("positions", []) if isinstance(account, Mapping) else []
        rows = [
            normalize_holding(position, account_hash, account_number)
            for position in positions
            if isinstance(position, Mapping)
        ]
        return pl.DataFrame(rows)

    def quotes(self, symbols: list[str] | tuple[str, ...] | str) -> JsonObject:
        """Return raw Schwab quote payload for one or more symbols."""

        symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
        payload = self._get_json(
            "/marketdata/v1/quotes", params={"symbols": ",".join(symbol_list)}
        )
        if not isinstance(payload, dict):
            raise SchwabProviderError("Schwab quotes payload was not a JSON object.")
        return payload

    def historical_prices(
        self,
        symbol: str,
        *,
        period_type: str = "day",
        period: int = 10,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> JsonObject:
        """Return raw Schwab historical price data for ``symbol``."""

        payload = self._get_json(
            f"/marketdata/v1/pricehistory/{symbol.upper()}",
            params={
                "periodType": period_type,
                "period": period,
                "frequencyType": frequency_type,
                "frequency": frequency,
            },
        )
        if not isinstance(payload, dict):
            raise SchwabProviderError(
                "Schwab historical-price payload was not a JSON object."
            )
        return payload

    @property
    def timeout(self) -> float:
        """Configured timeout passed to every transport call."""

        return self.timeout_seconds or self.settings.schwab_api_timeout_seconds

    def _get_json(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> JsonObject | list[Any]:
        return self._request_json("GET", path, params=params)

    def _request_json(
        self,
        method: str,
        path_or_url: str,
        *,
        params: Mapping[str, Any] | None = None,
        data: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        retry_refresh: bool = True,
    ) -> JsonObject | list[Any]:
        url = self._url(path_or_url, params)
        request = Request(url, data=data, headers=dict(headers or {}), method=method)
        if "Authorization" not in request.headers:
            request.add_header("Authorization", f"Bearer {self._access_token()}")
        request.add_header("Accept", "application/json")
        try:
            with self.transport(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 401 and retry_refresh:
                self.refresh_access_token()
                return self._request_json(
                    method,
                    path_or_url,
                    params=params,
                    data=data,
                    headers=headers,
                    retry_refresh=False,
                )
            raise SchwabProviderError(
                f"Schwab API request failed with HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise SchwabProviderError(
                "Schwab API request failed with a network error."
            ) from exc
        payload = json.loads(raw.decode() or "{}")
        if not isinstance(payload, dict | list):
            raise SchwabProviderError(
                "Schwab API response was not a JSON object or list."
            )
        return payload

    def refresh_access_token(self) -> None:
        """Refresh access token using the persisted refresh token."""

        tokens = self._load_tokens()
        refresh_token = self._required_str(tokens.get("refresh_token"), "refresh_token")
        client_id = self._required_str(
            self.settings.schwab_client_id, "schwab_client_id"
        )
        client_secret = self._required_str(
            self.settings.schwab_client_secret, "schwab_client_secret"
        )
        basic_token = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        body = urlencode(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        ).encode()
        payload = self._request_json(
            "POST",
            self.token_url,
            data=body,
            headers={
                "Authorization": f"Basic {basic_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            retry_refresh=False,
        )
        if not isinstance(payload, dict) or not payload.get("access_token"):
            raise SchwabAuthError(
                "Schwab token refresh did not return an access token."
            )
        merged = {**tokens, **payload, "refreshed_at": datetime.now(UTC).isoformat()}
        self._write_tokens(merged)
        self._tokens = merged

    def _access_token(self) -> str:
        tokens = self._load_tokens()
        if self._is_expired(tokens):
            self.refresh_access_token()
            tokens = self._load_tokens()
        return self._required_str(tokens.get("access_token"), "access_token")

    def _load_tokens(self) -> JsonObject:
        if self._tokens is not None:
            return self._tokens
        try:
            payload = json.loads(Path(self.settings.schwab_token_path).read_text())
        except FileNotFoundError as exc:
            raise SchwabProviderError("Schwab token file is missing.") from exc
        if not isinstance(payload, dict):
            raise SchwabProviderError("Schwab token file was not a JSON object.")
        self._tokens = payload
        return payload

    def _write_tokens(self, tokens: Mapping[str, Any]) -> None:
        token_path = Path(self.settings.schwab_token_path)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(token_path, flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as token_file:
            json.dump(dict(tokens), token_file, indent=2, sort_keys=True)
            token_file.write("\n")
        os.chmod(token_path, 0o600)

    def _is_expired(self, tokens: Mapping[str, Any]) -> bool:
        expires_at = tokens.get("expires_at")
        if not isinstance(expires_at, str):
            return False
        try:
            return datetime.fromisoformat(expires_at) <= datetime.now(UTC) + timedelta(
                seconds=30
            )
        except ValueError:
            return False

    def _url(self, path_or_url: str, params: Mapping[str, Any] | None) -> str:
        url = (
            path_or_url
            if path_or_url.startswith("http")
            else f"{self.base_url.rstrip('/')}/{path_or_url.lstrip('/')}"
        )
        if params:
            return f"{url}?{urlencode(params)}"
        return url

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _required_str(value: Any, name: str) -> str:
        if value is None or str(value) == "":
            raise SchwabProviderError(f"Missing required Schwab value: {name}")
        return str(value)


def normalize_holding(
    position: Mapping[str, Any], account_hash: str, account_number: Any = None
) -> JsonObject:
    """Normalize a Schwab position row into internal portfolio fields."""

    instrument = position.get("instrument", {})
    if not isinstance(instrument, Mapping):
        instrument = {}
    quantity = _number(
        position.get("longQuantity"),
        position.get("shortQuantity"),
        position.get("quantity"),
    )
    market_value = _number(position.get("marketValue"))
    current_price = _number(
        position.get("currentPrice"), instrument.get("currentPrice")
    )
    cost_basis = _number(position.get("costBasis"), position.get("averagePrice"))
    average_price = _number(position.get("averagePrice"))
    unrealized_pnl = _number(
        position.get("currentDayProfitLoss"), position.get("unrealizedProfitLoss")
    )
    pnl_percent = _number(
        position.get("currentDayProfitLossPercentage"),
        position.get("unrealizedProfitLossPercentage"),
    )
    return {
        "account_hash": account_hash,
        "masked_account_label": mask_account_label(account_number),
        "symbol": _string(instrument.get("symbol")),
        "asset_type": _string(instrument.get("assetType")),
        "quantity": quantity,
        "market_value": market_value,
        "current_price": current_price,
        "cost_basis": cost_basis,
        "average_price": average_price,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_percent": pnl_percent,
    }


def mask_account_label(account_number: Any) -> str:
    """Return a display-safe account label with only the final four digits."""

    if account_number is None or str(account_number).strip() == "":
        return "Account ****"
    digits = "".join(ch for ch in str(account_number) if ch.isdigit())
    suffix = digits[-4:] if digits else "****"
    return f"Account ****{suffix}"


def _number(*values: Any) -> float | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _string(value: Any) -> str | None:
    return None if value is None else str(value)
