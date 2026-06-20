"""Monitoring dashboard for Gooberberg Terminal."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from quant_platform.monitoring.service import MonitoringService


@st.cache_data(ttl=15)
def _summary() -> dict[str, Any]:
    return MonitoringService().summary().jsonable()


def _dataframe_or_info(rows: list[dict[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info(empty_message)


st.set_page_config(page_title="Monitoring", page_icon="🛰️", layout="wide")
st.title("Monitoring")
st.caption("Model operations overview for deployment readiness and trading status.")

summary = _summary()
st.info(
    "Deployment, online prediction serving, paper trading, and live trading are "
    "not implemented. This page surfaces catalog-backed readiness signals and "
    "clearly marks placeholders."
)
st.caption(f"Generated at: {summary['generated_at']}")

st.subheader("Active models")
active_models = summary["active_models"]
if active_models:
    model_cols = st.columns(4)
    model_cols[0].metric("Registered models", len(active_models))
    model_cols[1].metric(
        "Deployment implemented",
        sum(
            1
            for row in active_models
            if row.get("deployment_status") != "not_implemented"
        ),
    )
    model_cols[2].metric(
        "Serving implemented",
        sum(
            1 for row in active_models if row.get("serving_status") != "not_implemented"
        ),
    )
    model_cols[3].metric(
        "With artifacts", sum(1 for row in active_models if row.get("artifact_uri"))
    )
_dataframe_or_info(active_models, "No model definitions are registered yet.")

st.subheader("Recent predictions")
_dataframe_or_info(
    summary["recent_predictions"],
    (
        "No recent predictions are available because online prediction logging "
        "is not implemented."
    ),
)

st.subheader("Data freshness")
_dataframe_or_info(
    summary["data_freshness"],
    "No ingestion manifests or coverage rows are available for freshness checks yet.",
)

st.subheader("Drift checks")
_dataframe_or_info(
    summary["drift_checks"],
    "No drift checks are available because drift detection is not implemented.",
)

st.subheader("Latency")
latency = summary["latency"]
latency_cols = st.columns(4)
latency_cols[0].metric("Queued/running jobs", latency.get("queued_or_running_jobs", 0))
latency_cols[1].metric("Completed jobs", latency.get("completed_jobs", 0))
latency_cols[2].metric("Failed jobs", latency.get("failed_jobs", 0))
latency_cols[3].metric("Prediction latency", latency.get("prediction_latency_status"))
st.caption(latency.get("detail", ""))

st.subheader("Live/paper-trading status")
status_rows = [
    {"mode": mode.replace("_", " ").title(), **status}
    for mode, status in summary["trading_status"].items()
]
_dataframe_or_info(status_rows, "Trading status is unavailable.")
st.warning(
    "Live trading is not implemented. Do not use this application for live orders."
)
