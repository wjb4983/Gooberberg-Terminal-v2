"""Experiment queue and monitoring page for Gooberberg Terminal."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from quant_platform.common.enums import TaskType as JobTaskType
from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, experiment_metrics
from quant_platform.experiments.queueing import build_training_experiment_payload
from quant_platform.training.schemas import LossName, OptimizerName, TaskType


def _catalog() -> MetadataCatalog:
    catalog = MetadataCatalog(get_settings().catalog_db_path)
    catalog.create_all()
    return catalog


def _rows(table_name: str) -> list[dict[str, Any]]:
    return [dict(row) for row in _catalog().list_rows(table_name)]


def _label(row: dict[str, Any]) -> str:
    return f"#{row['id']} · {row['name']} v{row.get('version', '1')}"


def _feature_label(row: dict[str, Any]) -> str:
    feature_count = len(row.get("features") or [])
    return f"#{row['id']} · {row['name']} ({feature_count} features)"


def _model_label(row: dict[str, Any]) -> str:
    version = row.get("version", "1")
    model_type = row.get("model_type")
    return f"#{row['id']} · {row['name']} v{version} · {model_type}"


def _metrics(experiment_id: int) -> list[dict[str, Any]]:
    catalog = _catalog()
    with catalog.engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                experiment_metrics.select().where(
                    experiment_metrics.c.experiment_id == experiment_id
                )
            )
            .mappings()
            .all()
        ]


st.set_page_config(page_title="Experiments", page_icon="🧪", layout="wide")
st.title("Experiments")
st.caption("Configure supervised training experiments, queue jobs, and inspect status.")

datasets = _rows("dataset_definitions")
feature_sets = _rows("feature_sets")
models = _rows("model_definitions")

if not datasets or not models:
    st.warning("Create a dataset and model definition before queueing experiments.")

with st.form("queue_experiment"):
    st.subheader("Experiment inputs")
    name = st.text_input("Experiment name", value="baseline-training-run")
    left, right = st.columns(2)
    with left:
        dataset = (
            st.selectbox(
                "Dataset",
                datasets,
                format_func=_label,
                disabled=not datasets,
            )
            if datasets
            else None
        )
        compatible_feature_sets = [
            row
            for row in feature_sets
            if dataset is None or row.get("dataset_id") in {None, dataset["id"]}
        ]
        feature_set = (
            st.selectbox(
                "Feature set",
                compatible_feature_sets,
                format_func=_feature_label,
                disabled=not compatible_feature_sets,
            )
            if compatible_feature_sets
            else None
        )
        model = (
            st.selectbox(
                "Model definition",
                models,
                format_func=_model_label,
                disabled=not models,
            )
            if models
            else None
        )
        task_type = st.selectbox(
            "Task type", list(TaskType), format_func=lambda value: value.value
        )
    with right:
        target_name = st.text_input("Target/label name", value="forward_return")
        target_horizon = st.number_input("Target horizon", min_value=1, value=1, step=1)
        target_expression = st.text_input(
            "Target expression", value="weighted_feature_sum"
        )

    st.subheader("Split")
    split_cols = st.columns(3)
    with split_cols[0]:
        train_start = st.date_input("Train start", value=date(2024, 1, 1))
        train_end = st.date_input("Train end", value=date(2024, 3, 31))
    with split_cols[1]:
        validation_start = st.date_input("Validation start", value=date(2024, 4, 1))
        validation_end = st.date_input("Validation end", value=date(2024, 4, 30))
    with split_cols[2]:
        test_start = st.date_input("Test start", value=date(2024, 5, 1))
        test_end = st.date_input("Test end", value=date(2024, 5, 31))

    st.subheader("Training parameters")
    param_cols = st.columns(4)
    with param_cols[0]:
        epochs = st.number_input("Epochs", min_value=1, value=2, step=1)
        batch_size = st.number_input("Batch size", min_value=1, value=16, step=1)
    with param_cols[1]:
        optimizer = st.selectbox(
            "Optimizer", list(OptimizerName), format_func=lambda value: value.value
        )
        learning_rate = st.number_input(
            "Learning rate", min_value=0.000001, value=0.001, format="%.6f"
        )
    with param_cols[2]:
        loss_function = st.selectbox(
            "Loss function", list(LossName), format_func=lambda value: value.value
        )
        seed = st.number_input("Seed", value=7, step=1)
    with param_cols[3]:
        sequence_length = st.number_input(
            "Sequence length", min_value=1, value=8, step=1
        )
        hidden_size = st.number_input("Hidden size", min_value=1, value=16, step=1)

    payload: dict[str, Any] = {}
    if dataset is not None and model is not None:
        payload = build_training_experiment_payload(
            experiment_name=name,
            dataset=dataset,
            model=model,
            feature_set=feature_set,
            task_type=task_type,
            target={
                "name": target_name,
                "horizon": int(target_horizon),
                "expression": target_expression,
            },
            split={
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": test_start,
                "test_end": test_end,
            },
            training={
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "optimizer": optimizer,
                "learning_rate": float(learning_rate),
                "loss_function": loss_function,
                "sequence_length": int(sequence_length),
                "hidden_size": int(hidden_size),
                "seed": int(seed),
            },
        )
    with st.expander("Queued training payload preview", expanded=False):
        st.json(payload)
    submitted = st.form_submit_button(
        "Queue training job", disabled=not datasets or not models
    )

if submitted:
    catalog = _catalog()
    experiment_id = catalog.insert_row(
        "experiments",
        {
            "name": name.strip(),
            "status": "queued",
            "model_id": payload["model_id"],
            "dataset_id": payload["dataset_id"],
            "feature_set_id": payload["feature_set_id"],
            "parameters": {
                "task_type": payload["task_type"],
                "target": payload["target"],
                "split": payload["split"],
                "training": payload["training"],
            },
            "metadata": {"queued_payload": payload},
        },
    )
    payload["experiment_id"] = experiment_id
    job_id = catalog.insert_row(
        "jobs",
        {"job_type": JobTaskType.TRAIN.value, "status": "queued", "payload": payload},
    )
    st.success(f"Queued training job #{job_id} for experiment #{experiment_id}.")

st.subheader("Experiment status")
experiments = _rows("experiments")
if experiments:
    st.dataframe(
        pd.DataFrame(experiments)[
            [
                "id",
                "name",
                "status",
                "dataset_id",
                "feature_set_id",
                "model_id",
                "created_at",
                "started_at",
                "completed_at",
            ]
        ],
        use_container_width=True,
    )
    selected = st.selectbox(
        "Inspect experiment",
        experiments,
        format_func=lambda row: f"#{row['id']} · {row['name']} · {row['status']}",
    )
    st.json(
        {
            "parameters": selected.get("parameters") or {},
            "metadata": selected.get("metadata") or {},
        }
    )
    artifact_links = (
        (selected.get("metadata") or {}).get("artifacts")
        or (selected.get("metadata") or {}).get("artifact_links")
        or {}
    )
    metric_rows = _metrics(int(selected["id"]))
    if metric_rows:
        st.subheader("Metrics")
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True)
    else:
        st.info("No metrics have been logged for this experiment yet.")
    if artifact_links:
        st.subheader("Artifacts")
        for label, uri in artifact_links.items():
            st.markdown(f"- [{label}]({uri})")
    else:
        st.info("No log or artifact links are available yet.")
else:
    st.info("No experiments have been queued yet.")
