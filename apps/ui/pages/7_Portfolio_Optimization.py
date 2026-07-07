"""Portfolio optimization workspace for Gooberberg Terminal."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from apps.ui.portfolio_page_helpers import (
    BENCHMARK_OPTIONS,
    DEFAULT_LOOKBACK,
    account_options,
    currency,
    percent,
    portfolio_summary,
    setup_issues,
    show_setup_guidance,
)

from quant_platform.config import get_settings
from quant_platform.portfolio.metrics import LOOKBACKS

BASELINE_STRATEGIES = [
    "Current allocation",
    "Equal weight",
    "Market-cap proxy",
    "Minimum volatility",
    "Risk parity",
]


def _selected_account_hashes(
    accounts: list[dict[str, str]], account_label: str
) -> tuple[str, ...]:
    if account_label == "All accounts":
        return tuple(account["account_hash"] for account in accounts)
    return tuple(
        account["account_hash"]
        for account in accounts
        if account["label"] == account_label
    )


def _format_placeholder_allocations(rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    for column in ["current_weight", "placeholder_target_weight", "placeholder_delta"]:
        if column in table:
            table[column] = table[column].map(percent)
    if "market_value" in table:
        table["market_value"] = table["market_value"].map(currency)
    return table


def _placeholder_results(
    summary: dict[str, Any], selected_strategies: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allocation_rows = summary.get("allocation_by_symbol") or []
    if not allocation_rows:
        allocation_rows = [
            {"symbol": "SPY", "weight": 0.50, "market_value": 50_000},
            {"symbol": "QQQ", "weight": 0.30, "market_value": 30_000},
            {"symbol": "BIL", "weight": 0.20, "market_value": 20_000},
        ]

    symbol_count = max(len(allocation_rows), 1)
    target_weight = 1 / symbol_count
    allocation_table = []
    for row in allocation_rows:
        current_weight = float(row.get("weight") or 0)
        allocation_table.append(
            {
                "symbol": row.get("symbol", "UNKNOWN"),
                "market_value": row.get("market_value"),
                "current_weight": current_weight,
                "placeholder_target_weight": target_weight,
                "placeholder_delta": target_weight - current_weight,
                "label": "Placeholder result",
            }
        )

    strategy_table = pd.DataFrame(
        [
            {
                "baseline_strategy": strategy,
                "placeholder_return": 0.07 + (index * 0.005),
                "placeholder_volatility": 0.15 - (index * 0.006),
                "placeholder_sharpe": round(0.45 + (index * 0.04), 2),
                "label": "Placeholder result",
            }
            for index, strategy in enumerate(selected_strategies)
        ]
    )
    if not strategy_table.empty:
        strategy_table["placeholder_return"] = strategy_table[
            "placeholder_return"
        ].map(percent)
        strategy_table["placeholder_volatility"] = strategy_table[
            "placeholder_volatility"
        ].map(percent)

    return _format_placeholder_allocations(allocation_table), strategy_table


st.set_page_config(page_title="Portfolio Optimization", page_icon="📈", layout="wide")
st.title("Portfolio Optimization")
st.caption(
    "Compact portfolio optimization controls and placeholder outputs until "
    "optimization backend hooks are available."
)

setup_issues_list = setup_issues()
if setup_issues_list:
    show_setup_guidance(setup_issues_list)
    st.stop()

try:
    accounts = account_options()
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

settings = get_settings()
configured_lookbacks = [
    lookback.upper() for lookback in settings.schwab_supported_lookback_windows
]
lookbacks = [lookback for lookback in LOOKBACKS if lookback in configured_lookbacks]
if not lookbacks:
    lookbacks = list(LOOKBACKS)

with st.expander("Optimization controls", expanded=True):
    account_labels = ["All accounts", *[account["label"] for account in accounts]]
    top_left, top_middle, top_right = st.columns([2, 1, 1])
    account_label = top_left.selectbox(
        "Account scope",
        account_labels,
        key="portfolio_optimization_account_label",
    )
    benchmark = top_middle.selectbox(
        "Benchmark symbol",
        BENCHMARK_OPTIONS,
        index=BENCHMARK_OPTIONS.index(settings.schwab_default_benchmark)
        if settings.schwab_default_benchmark in BENCHMARK_OPTIONS
        else 0,
        key="portfolio_optimization_benchmark_symbol",
    )
    lookback = top_right.selectbox(
        "Lookback window",
        lookbacks,
        index=lookbacks.index(DEFAULT_LOOKBACK) if DEFAULT_LOOKBACK in lookbacks else 0,
        key="portfolio_optimization_lookback_window",
    )

    select_all = st.checkbox(
        "Select all baseline strategies",
        value=True,
        key="portfolio_optimization_select_all_baselines",
    )
    default_strategies = BASELINE_STRATEGIES if select_all else [BASELINE_STRATEGIES[0]]
    selected_strategies = st.multiselect(
        "Baseline strategies",
        BASELINE_STRATEGIES,
        default=default_strategies,
        key="portfolio_optimization_baseline_strategies",
    )
    run_requested = st.button("Run optimization", type="primary")

selected_hashes = _selected_account_hashes(accounts, account_label)

if run_requested and not selected_strategies:
    st.warning("Select at least one baseline strategy before running optimization.")
    st.stop()

try:
    summary = portfolio_summary(selected_hashes, benchmark)
except Exception as exc:  # noqa: BLE001 - page-level guidance for setup/runtime issues
    st.error(f"Unable to refresh portfolio data: {exc}")
    st.info("Confirm Schwab tokens are current, then run optimization again.")
    st.stop()

metadata = summary.get("metadata") or {}
st.caption(
    f"Scope: {account_label} | Benchmark: {benchmark} | Lookback: {lookback} | "
    f"Refreshed at: {metadata.get('refreshed_at', 'unknown')}"
)

if metadata.get("stale_data"):
    st.warning("Some portfolio data is served from cache; see warnings for details.")
for warning in summary.get("warnings") or []:
    st.warning(warning.get("message", "Portfolio warning."))

if run_requested:
    st.success(
        "Placeholder result: optimization backend hooks are not available yet, "
        "so these tables use current portfolio data plus mock target weights."
    )
else:
    st.info(
        "Placeholder result preview: click Run optimization to refresh the mock "
        "optimization tables with the selected controls."
    )

allocation_table, strategy_table = _placeholder_results(summary, selected_strategies)
left, right = st.columns([2, 1])
with left:
    st.subheader("Target allocation")
    st.dataframe(allocation_table, width="stretch")
with right:
    st.subheader("Baseline comparison")
    st.dataframe(strategy_table, width="stretch")

with st.expander("Inputs used", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {"input": "Account scope", "value": account_label},
                {"input": "Benchmark symbol", "value": benchmark},
                {"input": "Lookback window", "value": lookback},
                {
                    "input": "Baseline strategies",
                    "value": ", ".join(selected_strategies) or "None selected",
                },
                {"input": "Result label", "value": "Placeholder result"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )
