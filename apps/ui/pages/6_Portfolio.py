"""Schwab portfolio dashboard for Gooberberg Terminal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_platform.config import get_settings
from quant_platform.data.providers.schwab import SchwabProvider, mask_account_label
from quant_platform.portfolio.metrics import LOOKBACKS
from quant_platform.portfolio.service import PortfolioService

BENCHMARK_OPTIONS = ["SPY", "QQQ", "IWM", "DIA"]
DEFAULT_LOOKBACK = "1Y"


def _currency(value: Any) -> str:
    if value is None:
        return "—"
    return f"${float(value):,.2f}"


def _percent(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value) * 100:.2f}%"


def _setup_issues() -> list[str]:
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


def _show_setup_guidance(issues: list[str]) -> None:
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
def _account_options() -> list[dict[str, str]]:
    return [
        {"label": account.masked_account_label, "account_hash": account.account_hash}
        for account in SchwabProvider().account_numbers()
    ]


@st.cache_data(ttl=60, show_spinner=False)
def _portfolio_summary(
    account_hashes: tuple[str, ...], benchmark_symbol: str
) -> dict[str, Any]:
    return PortfolioService().summary(
        account_hashes=account_hashes or None,
        benchmark_symbol=benchmark_symbol,
    )


def _render_metrics(totals: dict[str, Any]) -> None:
    columns = st.columns(4)
    columns[0].metric("Total market value", _currency(totals.get("total_value")))
    columns[1].metric("Cash value", _currency(totals.get("cash_value")))
    columns[2].metric("Securities value", _currency(totals.get("securities_value")))
    columns[3].metric("Unrealized PnL", _currency(totals.get("unrealized_pnl")))


def _render_allocations(summary: dict[str, Any]) -> None:
    symbol_rows = summary.get("allocation_by_symbol") or []
    asset_rows = summary.get("allocation_by_asset_type") or []
    left, right = st.columns(2)
    with left:
        st.subheader("Allocation by symbol")
        if symbol_rows:
            df = pd.DataFrame(symbol_rows).set_index("symbol")
            st.bar_chart(df["weight"])
            st.dataframe(
                _format_allocation_table(pd.DataFrame(symbol_rows)),
                width="stretch",
            )
        else:
            st.info("No symbol allocation is available yet.")
    with right:
        st.subheader("Allocation by asset type")
        if asset_rows:
            df = pd.DataFrame(asset_rows).set_index("asset_type")
            st.bar_chart(df["weight"])
            st.dataframe(
                _format_allocation_table(pd.DataFrame(asset_rows)),
                width="stretch",
            )
        else:
            st.info("No asset-type allocation is available yet.")


def _format_allocation_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    if "market_value" in formatted:
        formatted["market_value"] = formatted["market_value"].map(_currency)
    if "weight" in formatted:
        formatted["weight"] = formatted["weight"].map(_percent)
    return formatted


def _render_holdings(rows: list[dict[str, Any]]) -> None:
    st.subheader("Holdings")
    if not rows:
        st.info("No holdings were returned for the selected account scope.")
        return
    table = pd.DataFrame(rows)
    safe_columns = [
        "masked_account_label",
        "symbol",
        "asset_type",
        "quantity",
        "market_value",
        "current_price",
        "cost_basis",
        "average_price",
        "unrealized_pnl",
        "unrealized_pnl_percent",
    ]
    table = table[[column for column in safe_columns if column in table.columns]]
    table = table.rename(columns={"masked_account_label": "account"})
    if "account" in table:
        table["account"] = table["account"].map(mask_account_label)
    for column in [
        "market_value",
        "current_price",
        "cost_basis",
        "average_price",
        "unrealized_pnl",
    ]:
        if column in table:
            table[column] = table[column].map(_currency)
    if "unrealized_pnl_percent" in table:
        table["unrealized_pnl_percent"] = table["unrealized_pnl_percent"].map(_percent)
    st.dataframe(table, width="stretch")


def _render_lookback_metrics(summary: dict[str, Any], selected_lookback: str) -> None:
    st.subheader("Lookback metrics")
    metrics = summary.get("lookback_metrics") or {}
    normalized_lookback = selected_lookback.upper()
    selected = metrics.get(normalized_lookback)
    if not selected:
        st.info(f"No metrics are available for the {normalized_lookback} lookback.")
        return
    row = {
        "lookback": selected.get("lookback", normalized_lookback),
        "total_return": _percent(selected.get("total_return")),
        "annualized_volatility": _percent(selected.get("annualized_volatility")),
        "sharpe_ratio": selected.get("sharpe_ratio"),
        "beta": selected.get("beta"),
        "max_drawdown": _percent(selected.get("max_drawdown")),
    }
    st.dataframe(pd.DataFrame([row]), width="stretch")


def _render_optimization_link() -> None:
    st.divider()
    st.subheader("Portfolio Optimization")
    st.caption(
        "Use the selected account scope and benchmark context to explore optimized "
        "allocation ideas."
    )
    if hasattr(st, "page_link"):
        st.page_link(
            "pages/7_Portfolio_Optimization.py",
            label="Portfolio Optimization",
            icon="📈",
        )
    else:
        st.markdown("[📈 Portfolio Optimization](7_Portfolio_Optimization)")


st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")
st.title("Portfolio")
st.caption("Schwab-backed account, allocation, holding, and lookback analytics.")

setup_issues = _setup_issues()
if setup_issues:
    _show_setup_guidance(setup_issues)
    st.stop()

try:
    accounts = _account_options()
except Exception as exc:  # noqa: BLE001 - page-level guidance for setup/runtime issues
    st.error(f"Unable to load Schwab accounts: {exc}")
    st.info(
        "Verify the token file exists, is readable, and contains valid Schwab "
        "OAuth tokens."
    )
    st.stop()

if not accounts:
    st.info("No Schwab accounts were returned for the configured token file.")
    st.stop()

account_labels = ["All accounts", *[account["label"] for account in accounts]]
account_label = st.selectbox("Account", account_labels, key="portfolio_account_label")
benchmark = st.selectbox(
    "Benchmark",
    BENCHMARK_OPTIONS,
    index=BENCHMARK_OPTIONS.index(get_settings().schwab_default_benchmark)
    if get_settings().schwab_default_benchmark in BENCHMARK_OPTIONS
    else 0,
    key="portfolio_benchmark_symbol",
)
configured_lookbacks = [
    lookback.upper() for lookback in get_settings().schwab_supported_lookback_windows
]
lookbacks = [lookback for lookback in LOOKBACKS if lookback in configured_lookbacks]
if not lookbacks:
    lookbacks = list(LOOKBACKS)
lookback = st.selectbox(
    "Lookback",
    lookbacks,
    index=lookbacks.index(DEFAULT_LOOKBACK) if DEFAULT_LOOKBACK in lookbacks else 0,
    key="portfolio_lookback_window",
)
if st.button("Refresh", type="primary"):
    _account_options.clear()
    _portfolio_summary.clear()
    st.rerun()

selected_hashes = tuple(
    account["account_hash"] for account in accounts if account["label"] == account_label
)
if account_label == "All accounts":
    selected_hashes = tuple(account["account_hash"] for account in accounts)

try:
    summary = _portfolio_summary(selected_hashes, benchmark)
except Exception as exc:  # noqa: BLE001 - page-level guidance for setup/runtime issues
    st.error(f"Unable to refresh portfolio data: {exc}")
    st.info("Confirm Schwab tokens are current, then use Refresh to retry.")
    st.stop()

metadata = summary.get("metadata") or {}
st.caption(
    f"Refreshed at: {metadata.get('refreshed_at', 'unknown')} | "
    f"Holdings as of: {metadata.get('holdings_refreshed_at', 'unknown')} | "
    f"Prices as of: {metadata.get('prices_refreshed_at', 'unknown')} | "
    f"Benchmark: {benchmark}"
)
if metadata.get("stale_data"):
    st.warning("Some portfolio data is served from cache; see warnings for details.")
for warning in summary.get("warnings") or []:
    st.warning(warning.get("message", "Portfolio warning."))

_render_metrics(summary.get("totals") or {})
_render_allocations(summary)
_render_holdings(summary.get("holdings") or [])
_render_lookback_metrics(summary, lookback)
_render_optimization_link()
