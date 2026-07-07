"""Shared Streamlit helpers for Schwab portfolio pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from quant_platform.config import get_settings
from quant_platform.data.providers.schwab import SchwabProvider
from quant_platform.portfolio.service import PortfolioService

BENCHMARK_OPTIONS = ["SPY", "QQQ", "IWM", "DIA"]
DEFAULT_LOOKBACK = "1Y"


def currency(value: Any) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def setup_issues() -> list[str]:
    settings = get_settings()
    issues: list[str] = []
    if not settings.schwab_client_id:
        issues.append(
            "Set `QUANT_PLATFORM_SCHWAB_CLIENT_ID` to your Schwab app client ID."
        )
    if not settings.schwab_client_secret:
        issues.append(
            "Set `QUANT_PLATFORM_SCHWAB_CLIENT_SECRET` to your Schwab app "
            "client secret."
        )
    token_path = Path(settings.schwab_token_path)
    if not token_path.exists():
        issues.append(
            "Create the Schwab OAuth token file at "
            f"`{token_path}` with the Schwab auth bootstrap flow."
        )
    return issues


def show_setup_guidance(issues: list[str]) -> None:
    st.warning("Schwab portfolio access is not configured yet.")
    st.markdown("Complete these quick-start tasks in order:")
    for index, issue in enumerate(issues, start=1):
        st.markdown(f"{index}. {issue}")
    st.info(
        "This page only stores display selections in Streamlit session state. "
        "Schwab secrets and OAuth token contents are read by the provider and are "
        "never copied into session state."
    )


@st.cache_data(ttl=60, show_spinner=False)
def account_options() -> list[dict[str, str]]:
    return [
        {"label": account.masked_account_label, "account_hash": account.account_hash}
        for account in SchwabProvider().account_numbers()
    ]


@st.cache_data(ttl=60, show_spinner=False)
def portfolio_summary(
    account_hashes: tuple[str, ...], benchmark_symbol: str
) -> dict[str, Any]:
    return PortfolioService().summary(
        account_hashes=account_hashes or None,
        benchmark_symbol=benchmark_symbol,
    )
