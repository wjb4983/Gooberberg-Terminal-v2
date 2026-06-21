"""Monitoring dashboard for Gooberberg Terminal."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from sqlalchemy import select

from quant_platform.common.enums import JobStatus
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, job_logs, jobs
from quant_platform.monitoring.service import MonitoringService


@st.cache_data(ttl=15)
def _summary() -> dict[str, Any]:
    return MonitoringService().summary().jsonable()


def _dataframe_or_info(rows: list[dict[str, Any]], empty_message: str) -> None:
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info(empty_message)


@st.cache_data(ttl=5)
def _job_board(limit: int = 50) -> dict[str, list[dict[str, Any]]]:
    catalog = MetadataCatalog(get_settings().catalog_db_path)
    catalog.create_all()
    with catalog.engine.connect() as connection:
        rows = [
            dict(row)
            for row in connection.execute(
                select(jobs).order_by(jobs.c.created_at.desc()).limit(limit)
            ).mappings()
        ]
        log_rows = [
            dict(row)
            for row in connection.execute(
                select(job_logs).order_by(
                    job_logs.c.created_at.desc(), job_logs.c.id.desc()
                )
            ).mappings()
        ]
    logs_by_job: dict[int, list[dict[str, Any]]] = {}
    for row in log_rows:
        logs_by_job.setdefault(int(row["job_id"]), []).append(row)
    for row in rows:
        payload = row.get("payload") or {}
        row["summary"] = (
            f"{row['job_type']} "
            f"{payload.get('dataset_name') or payload.get('model') or ''}"
        ).strip()
        row["latest_log"] = next(
            (
                log["message"]
                for log in logs_by_job.get(int(row["id"]), [])
                if log.get("message")
            ),
            "",
        )
        row["logs"] = list(reversed(logs_by_job.get(int(row["id"]), [])[:25]))
    return {
        "queued": [row for row in rows if row["status"] == JobStatus.QUEUED.value],
        "running": [row for row in rows if row["status"] == JobStatus.RUNNING.value],
        "finished": [
            row
            for row in rows
            if row["status"]
            in {
                JobStatus.SUCCEEDED.value,
                JobStatus.FAILED.value,
                JobStatus.CANCELLED.value,
            }
        ],
    }


def _render_job_group(title: str, rows: list[dict[str, Any]]) -> None:
    st.markdown(f"#### {title}")
    if not rows:
        st.info(f"No {title.lower()} jobs.")
        return
    table_rows = [
        {
            "id": row["id"],
            "type": row["job_type"],
            "status": row["status"],
            "summary": row["summary"],
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "completed_at": row.get("completed_at"),
            "latest_log": row.get("latest_log", ""),
            "error": row.get("error"),
        }
        for row in rows
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
    for row in rows:
        with st.expander(f"Job #{row['id']} logs and payload", expanded=False):
            st.json(row.get("payload") or {})
            logs = row.get("logs") or []
            if logs:
                st.dataframe(pd.DataFrame(logs), use_container_width=True)
            else:
                st.info("No logs have been recorded for this job yet.")


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

st.subheader("Job queue")
st.caption(
    "Refreshes every few seconds and shows queued work, actively running jobs, "
    "finished jobs, payload details, errors, and lifecycle logs."
)
job_board = _job_board()
job_cols = st.columns(3)
job_cols[0].metric("Queued", len(job_board["queued"]))
job_cols[1].metric("Running", len(job_board["running"]))
job_cols[2].metric("Finished", len(job_board["finished"]))
_render_job_group("Queued", job_board["queued"])
_render_job_group("Running", job_board["running"])
_render_job_group("Finished", job_board["finished"])

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
