"""Backtest queue and results page for Gooberberg Terminal."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_platform.backtesting.schemas import (
    BacktestConfig,
    OptionsExpirationHandling,
    PositionSizingMode,
)
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog


def _catalog() -> MetadataCatalog:
    catalog = MetadataCatalog(get_settings().catalog_db_path)
    catalog.create_all()
    return catalog


def _rows(table_name: str) -> list[dict[str, Any]]:
    return [dict(row) for row in _catalog().list_rows(table_name)]


def _asset_label(row: dict[str, Any]) -> str:
    return f"#{row['id']} · {row['name']} v{row.get('version', '1')}"


def _model_label(row: dict[str, Any]) -> str:
    version = row.get("version", "1")
    model_type = row.get("model_type")
    return f"#{row['id']} · {row['name']} v{version} · {model_type}"


def _strategy_options() -> list[dict[str, str]]:
    return [
        {"id": "moving_average_cross", "name": "Moving average cross (placeholder)"},
        {"id": "mean_reversion", "name": "Mean reversion (placeholder)"},
        {"id": "breakout", "name": "Breakout (placeholder)"},
    ]


def _read_artifact_table(path: str | None) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    return None


st.set_page_config(page_title="Backtests", page_icon="📈", layout="wide")
st.title("Backtests")
st.caption("Queue model or signal-strategy backtests and inspect results.")

models = _rows("model_definitions")
datasets = _rows("dataset_definitions")
backtests = _rows("backtests")

if not datasets:
    st.warning("Create a dataset before queueing backtests.")

with st.form("queue_backtest"):
    st.subheader("Backtest inputs")
    name = st.text_input("Backtest name", value="baseline-backtest")
    source_type = st.radio(
        "Signal source",
        ["Trained model", "Saved signal strategy"],
        horizontal=True,
    )

    left, right = st.columns(2)
    with left:
        selected_model = None
        selected_strategy = None
        if source_type == "Trained model":
            selected_model = (
                st.selectbox(
                    "Trained model / model definition",
                    models,
                    format_func=_model_label,
                    disabled=not models,
                )
                if models
                else None
            )
            if not models:
                st.info("No trained models or model definitions are registered yet.")
        else:
            selected_strategy = st.selectbox(
                "Saved signal strategy placeholder",
                _strategy_options(),
                format_func=lambda row: row["name"],
            )
        selected_dataset = (
            st.selectbox(
                "Dataset", datasets, format_func=_asset_label, disabled=not datasets
            )
            if datasets
            else None
        )
    with right:
        start = st.date_input("Start date", value=date(2024, 1, 1))
        end = st.date_input("End date", value=date(2024, 12, 31))
        initial_cash = st.number_input("Initial cash", min_value=1.0, value=100_000.0)
        annualization_factor = st.number_input(
            "Annualization factor", min_value=1, value=252, step=1
        )

    st.subheader("Execution assumptions")
    cols = st.columns(4)
    with cols[0]:
        price_column = st.text_input("Price column", value="close")
        signal_column = st.text_input("Signal column", value="signal")
        timestamp_column = st.text_input("Timestamp column", value="timestamp")
        symbol_column = st.text_input("Symbol column", value="symbol")
    with cols[1]:
        sizing_mode = st.selectbox(
            "Position sizing",
            list(PositionSizingMode),
            format_func=lambda value: value.value,
        )
        fixed_notional = st.number_input(
            "Fixed notional", min_value=1.0, value=10_000.0
        )
        max_gross_exposure = st.number_input(
            "Max gross exposure", min_value=0.01, value=1.0
        )
        target_weight_column = st.text_input(
            "Target weight column", value="target_weight"
        )
    with cols[2]:
        slippage_enabled = st.checkbox("Enable slippage")
        slippage_bps = st.number_input("Slippage bps", min_value=0.0, value=0.0)
        commission_enabled = st.checkbox("Enable commission")
        commission_per_trade = st.number_input(
            "Commission per trade", min_value=0.0, value=0.0
        )
        commission_bps = st.number_input("Commission bps", min_value=0.0, value=0.0)
    with cols[3]:
        bid_ask_enabled = st.checkbox("Use bid/ask fills")
        bid_column = st.text_input("Bid column", value="bid")
        ask_column = st.text_input("Ask column", value="ask")
        max_bars = st.number_input("Max holding bars (0 = off)", min_value=0, value=0)
        options_policy = st.selectbox(
            "Options expiration handling",
            list(OptionsExpirationHandling),
            format_func=lambda value: value.value,
        )

    config = BacktestConfig(
        name=name.strip() or "backtest",
        initial_cash=float(initial_cash),
        price_column=price_column,
        signal_column=signal_column,
        timestamp_column=timestamp_column,
        symbol_column=symbol_column,
        annualization_factor=int(annualization_factor),
        slippage={"enabled": slippage_enabled, "basis_points": float(slippage_bps)},
        commission={
            "enabled": commission_enabled,
            "per_trade": float(commission_per_trade),
            "basis_points": float(commission_bps),
        },
        bid_ask={
            "enabled": bid_ask_enabled,
            "bid_column": bid_column,
            "ask_column": ask_column,
        },
        position_sizing={
            "mode": sizing_mode,
            "fixed_notional": float(fixed_notional),
            "target_weight_column": target_weight_column,
        },
        exposure={"max_gross_exposure": float(max_gross_exposure)},
        holding_period={"max_bars": int(max_bars) or None},
        options_expiration_handling=options_policy,
    )
    payload = {
        "name": name.strip() or "backtest",
        "source_type": "model" if source_type == "Trained model" else "signal_strategy",
        "model_id": (selected_model or {}).get("id"),
        "signal_strategy": (selected_strategy or {}).get("id"),
        "dataset_id": (selected_dataset or {}).get("id"),
        "dataset_name": (selected_dataset or {}).get("name"),
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "config": config.jsonable(),
    }
    with st.expander("Queued backtest payload preview", expanded=False):
        st.json(payload)
    disabled = not datasets or (source_type == "Trained model" and not models)
    submitted = st.form_submit_button("Queue backtest", disabled=disabled)

if submitted:
    catalog = _catalog()
    backtest_id = catalog.insert_row(
        "backtests",
        {
            "experiment_id": None,
            "name": payload["name"],
            "status": "queued",
            "start_ts": pd.Timestamp(start).to_pydatetime(),
            "end_ts": pd.Timestamp(end).to_pydatetime(),
            "results": {},
            "metadata": payload,
        },
    )
    payload["backtest_id"] = backtest_id
    job_id = catalog.insert_row(
        "jobs", {"job_type": "backtest", "status": "queued", "payload": payload}
    )
    st.success(f"Queued backtest job #{job_id} for backtest #{backtest_id}.")
    backtests = _rows("backtests")

st.subheader("Backtest status")
if backtests:
    st.dataframe(
        pd.DataFrame(backtests)[
            ["id", "name", "status", "start_ts", "end_ts", "created_at"]
        ],
        use_container_width=True,
    )
    selected = st.selectbox(
        "Inspect backtest",
        backtests,
        format_func=lambda row: f"#{row['id']} · {row['name']} · {row['status']}",
    )
    results = selected.get("results") or {}
    metadata = selected.get("metadata") or {}
    st.json({"results": results, "metadata": metadata})

    metrics = results.get("metrics") or metadata.get("metrics") or {}
    if metrics:
        st.subheader("Metrics")
        metric_cols = st.columns(min(4, max(1, len(metrics))))
        for index, (metric_name, value) in enumerate(metrics.items()):
            metric_cols[index % len(metric_cols)].metric(metric_name, value)
    else:
        st.info("No metrics have been logged for this backtest yet.")

    files = results.get("files") or metadata.get("files") or {}
    equity_curve = _read_artifact_table(files.get("equity_curve"))
    trades = _read_artifact_table(files.get("trades"))

    st.subheader("Equity curve")
    if equity_curve is not None and not equity_curve.empty:
        chart_data = (
            equity_curve.set_index("timestamp")
            if "timestamp" in equity_curve
            else equity_curve
        )
        st.line_chart(chart_data[["equity"]] if "equity" in chart_data else chart_data)
        st.dataframe(equity_curve, use_container_width=True)
    else:
        st.info("No equity curve artifact is available yet.")

    st.subheader("Trade log")
    if trades is not None and not trades.empty:
        st.dataframe(trades, use_container_width=True)
    else:
        st.info("No trade log artifact is available yet.")
else:
    st.info("No backtests have been queued yet.")
