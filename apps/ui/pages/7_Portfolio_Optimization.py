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


def _format_decimal(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"


def _weight_summary(weights: dict[str, Any]) -> str:
    if not weights:
        return "—"
    sorted_weights = sorted(
        weights.items(), key=lambda item: abs(float(item[1] or 0)), reverse=True
    )
    summary_parts = [
        f"{symbol} {percent(weight)}" for symbol, weight in sorted_weights[:3]
    ]
    if len(sorted_weights) > 3:
        summary_parts.append(f"+{len(sorted_weights) - 3} more")
    return ", ".join(summary_parts)


def _warning_summary(warnings: list[Any], constraints: list[Any]) -> str:
    warning_count = len(warnings)
    constraint_count = len(constraints)
    if not warning_count and not constraint_count:
        return "None"
    parts = []
    if warning_count:
        parts.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
    if constraint_count:
        parts.append(
            f"{constraint_count} constraint{'s' if constraint_count != 1 else ''}"
        )
    return ", ".join(parts)


def _render_optimization_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Render compact optimization comparison rows plus optional details."""
    comparison_rows: list[dict[str, Any]] = []
    weight_detail_rows: list[dict[str, Any]] = []
    warning_detail_rows: list[dict[str, Any]] = []

    for result in results:
        strategy = str(result.get("strategy_name") or "Unnamed strategy")
        weights = result.get("target_weights") or {}
        warnings = result.get("warnings") or []
        constraints = result.get("constraints") or []
        is_placeholder = bool(result.get("placeholder"))
        status = "Placeholder" if is_placeholder else "Live"

        comparison_rows.append(
            {
                "strategy name": strategy,
                "target weights summary": _weight_summary(weights),
                "expected return": percent(result.get("expected_return")),
                "volatility": percent(result.get("volatility")),
                "Sharpe": _format_decimal(result.get("sharpe")),
                "max drawdown": percent(result.get("max_drawdown")),
                "turnover": percent(result.get("turnover")),
                "leverage": _format_decimal(result.get("leverage")),
                "warnings/constraints": _warning_summary(warnings, constraints),
                "placeholder status": status,
            }
        )

        for symbol, weight in sorted(weights.items()):
            weight_detail_rows.append(
                {
                    "strategy": strategy,
                    "symbol": symbol,
                    "target_weight": percent(weight),
                    "placeholder_status": status,
                }
            )
        for warning in warnings:
            warning_detail_rows.append(
                {
                    "strategy": strategy,
                    "type": "warning",
                    "message": warning,
                    "placeholder_status": status,
                }
            )
        for constraint in constraints:
            warning_detail_rows.append(
                {
                    "strategy": strategy,
                    "type": "constraint",
                    "message": constraint,
                    "placeholder_status": status,
                }
            )

    comparison_table = pd.DataFrame(comparison_rows)
    has_placeholder = any(result.get("placeholder") for result in results)
    st.subheader("Strategy comparison")
    if has_placeholder:
        st.caption(
            "Placeholder rows use mock metrics until backend hooks are connected."
        )
    st.dataframe(comparison_table, width="stretch", hide_index=True)

    if weight_detail_rows:
        with st.expander("Detailed target weights", expanded=False):
            st.dataframe(
                pd.DataFrame(weight_detail_rows), width="stretch", hide_index=True
            )
    if warning_detail_rows:
        with st.expander("Full warning and constraint lists", expanded=False):
            st.dataframe(
                pd.DataFrame(warning_detail_rows), width="stretch", hide_index=True
            )
    return comparison_table


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
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
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

    target_weights = {
        row["symbol"]: row["placeholder_target_weight"] for row in allocation_table
    }
    optimization_results = [
        {
            "strategy_name": strategy,
            "target_weights": target_weights,
            "expected_return": 0.07 + (index * 0.005),
            "volatility": 0.15 - (index * 0.006),
            "sharpe": round(0.45 + (index * 0.04), 2),
            "max_drawdown": -0.18 + (index * 0.01),
            "turnover": 0.12 + (index * 0.015),
            "leverage": 1.0,
            "warnings": [
                "Placeholder output: optimization backend hooks are not available yet."
            ],
            "constraints": ["Long-only placeholder targets", "Weights sum to 100%"],
            "placeholder": True,
        }
        for index, strategy in enumerate(selected_strategies)
    ]

    return _format_placeholder_allocations(allocation_table), optimization_results


st.set_page_config(page_title="Portfolio Optimization", page_icon="📈", layout="wide")
st.title("Portfolio Optimization")
st.caption("Compare baseline allocation strategies with concise setup guidance.")

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
    with st.expander("Advanced constraints", expanded=False):
        st.caption("Optional guardrails for future optimizer runs.")
        constraint_left, constraint_right = st.columns(2)
        long_only = constraint_left.checkbox(
            "Long-only targets",
            value=True,
            key="portfolio_optimization_long_only",
        )
        max_position_weight = constraint_right.slider(
            "Max position weight",
            min_value=0.05,
            max_value=1.0,
            value=0.35,
            step=0.05,
            format="%.2f",
            key="portfolio_optimization_max_position_weight",
        )
        max_turnover = st.slider(
            "Max turnover",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            format="%.2f",
            key="portfolio_optimization_max_turnover",
        )
        st.caption("If a constraint blocks a solution, loosen it and rerun.")

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

with st.expander("Backend status", expanded=False):
    st.warning("Optimizer backend is not connected yet.")
    st.markdown("1. Use current portfolio data for inputs.")
    st.markdown("2. Review placeholder targets before trading.")
    st.markdown("3. Connect optimizer hooks to enable live results.")

if run_requested:
    st.success("Run complete. Results below are placeholder optimizer outputs.")
else:
    st.info("Preview mode. Click Run optimization to refresh placeholder tables.")

allocation_table, optimization_results = _placeholder_results(
    summary, selected_strategies
)
_render_optimization_results(optimization_results)

with st.expander("Detailed strategy assumptions", expanded=False):
    st.caption("Placeholder assumptions used for the comparison table.")
    st.markdown("- Targets are equal-weight across displayed symbols.")
    st.markdown("- Metrics are mock values for UI validation.")
    st.markdown("- No trades are generated from this page yet.")

with st.expander("Target allocation preview", expanded=False):
    st.dataframe(allocation_table, width="stretch", hide_index=True)

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
                {"input": "Long-only targets", "value": "Yes" if long_only else "No"},
                {"input": "Max position weight", "value": percent(max_position_weight)},
                {"input": "Max turnover", "value": percent(max_turnover)},
                {"input": "Result label", "value": "Placeholder result"},
            ]
        ),
        width="stretch",
        hide_index=True,
    )
