"""Experiment queue and monitoring page for Gooberberg Terminal."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import streamlit as st
from apps.ui.experiment_context import (
    default_target_for_context,
    default_training_for_context,
    is_queueable_experiment_context,
)
from apps.ui.experiment_selection import (
    compatibility_messages as _compatibility_messages,
)
from apps.ui.experiment_selection import (
    compatible_model_options,
)
from apps.ui.experiment_selection import (
    feature_set_has_required_columns as _feature_set_has_required_columns,
)
from apps.ui.experiment_selection import (
    normalize_text as _normalize_text,
)
from apps.ui.experiment_selection import (
    required_model_columns as _required_model_columns,
)
from apps.ui.workflow_context import (
    ACTIVE_DATASET_ID_KEY,
    ACTIVE_MODEL_ID_KEY,
    context_from_dataset_row,
    context_from_model_row,
    merge_workflow_context,
    store_workflow_context,
)

from quant_platform.config import get_settings
from quant_platform.data.storage.catalog import MetadataCatalog, experiment_metrics
from quant_platform.experiments.queueing import (
    build_training_experiment_payload,
    create_and_enqueue_training_experiment,
)
from quant_platform.models import ModelType
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


def _active_index(rows: list[dict[str, Any]], session_key: str) -> int:
    active_id = st.session_state.get(session_key)
    for index, row in enumerate(rows):
        if row.get("id") == active_id:
            return index
    return 0


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
                index=_active_index(datasets, ACTIVE_DATASET_ID_KEY),
                format_func=_label,
                disabled=not datasets,
            )
            if datasets
            else None
        )
        dataset_context = context_from_dataset_row(dataset)
        if dataset is not None:
            store_workflow_context(st.session_state, dataset_context)

        show_incompatible_models = st.checkbox(
            "Show incompatible models",
            value=False,
            help=(
                "Advanced override: include models that do not match selected "
                "dataset metadata."
            ),
        )
        model_options = compatible_model_options(
            dataset, models, show_incompatible=show_incompatible_models
        )
        model = (
            st.selectbox(
                "Model definition",
                model_options,
                index=_active_index(model_options, ACTIVE_MODEL_ID_KEY),
                format_func=_model_label,
                disabled=not model_options,
            )
            if model_options
            else None
        )
        required_columns = _required_model_columns(model)
        compatible_feature_sets = [
            row
            for row in feature_sets
            if (dataset is None or row.get("dataset_id") in {None, dataset["id"]})
            and _feature_set_has_required_columns(row, required_columns)
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
        override_compatibility = st.checkbox(
            "Override compatibility checks",
            value=False,
            disabled=not show_incompatible_models,
            help=(
                "Expert override: allow queueing even when the selected "
                "dataset/model/feature-set combination fails required "
                "compatibility checks."
            ),
        )
    target_defaults = default_target_for_context(model)
    training_defaults = default_training_for_context(model)
    with right:
        model_type = _normalize_text(model.get("model_type") if model else None)
        if model_type == ModelType.REGIME_DETECTOR.value:
            st.caption("Regime detector target context")
            target_name = st.text_input(
                "Regime label column", value=target_defaults["name"]
            )
            target_horizon = st.number_input(
                "Regime target horizon",
                min_value=1,
                value=int(target_defaults["horizon"]),
                step=1,
            )
            target_expression = st.text_input(
                "Regime feature columns or discovery mode",
                value=target_defaults["expression"],
                help=(
                    "Use 'regime_classification' for labeled regimes or list the "
                    "feature columns used by unsupervised regime discovery."
                ),
            )
        elif model_type == ModelType.REGIME_SWITCHING.value:
            switching_defaults = (
                (model.get("metadata") or {}).get("regime_switching", {})
                if model
                else {}
            )
            st.caption("Regime-switching allocation target context")
            target_name = st.text_input(
                "Allocation / weight target column", value=target_defaults["name"]
            )
            target_horizon = st.number_input(
                "Allocation target horizon",
                min_value=1,
                value=int(target_defaults["horizon"]),
                step=1,
            )
            target_expression = st.text_input(
                "Signal column", value=target_defaults["expression"]
            )
            st.text_input(
                "Regime column",
                value=str(switching_defaults.get("regime_column", "regime")),
                disabled=True,
            )
        else:
            target_name = st.text_input(
                "Target/label name", value=target_defaults["name"]
            )
            target_horizon = st.number_input(
                "Target horizon",
                min_value=1,
                value=int(target_defaults["horizon"]),
                step=1,
            )
            target_expression = st.text_input(
                "Target expression", value=target_defaults["expression"]
            )
        task_type = st.selectbox(
            "Task type",
            list(TaskType),
            index=list(TaskType).index(training_defaults["task_type"]),
            format_func=lambda value: value.value,
        )

    compatibility_reasons, compatibility_warnings = _compatibility_messages(
        dataset, model
    )
    missing_feature_columns = bool(required_columns and feature_set is None)
    compatibility_blockers = list(compatibility_warnings)
    if missing_feature_columns:
        compatibility_blockers.append(
            (
                "Warning: selected model requires feature columns that no available "
                "feature set provides: "
            )
            + ", ".join(sorted(required_columns))
            + "."
        )

    st.subheader("Compatibility")
    if dataset is not None and model is not None:
        for reason in compatibility_reasons:
            st.success(reason)
        for warning in compatibility_blockers:
            st.warning(warning)
        if compatibility_blockers and not override_compatibility:
            st.error(
                "Queueing is disabled until compatibility blockers are resolved "
                "or explicitly overridden."
            )
    else:
        st.info("Select a dataset and model to see compatibility details.")

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
        epochs = st.number_input(
            "Epochs", min_value=1, value=int(training_defaults["epochs"]), step=1
        )
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            value=int(training_defaults["batch_size"]),
            step=1,
        )
    with param_cols[1]:
        optimizer = st.selectbox(
            "Optimizer",
            list(OptimizerName),
            index=list(OptimizerName).index(training_defaults["optimizer"]),
            format_func=lambda value: value.value,
        )
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.000001,
            value=float(training_defaults["learning_rate"]),
            format="%.6f",
        )
    with param_cols[2]:
        loss_function = st.selectbox(
            "Loss function",
            list(LossName),
            index=list(LossName).index(training_defaults["loss_function"]),
            format_func=lambda value: value.value,
        )
        seed = st.number_input("Seed", value=int(training_defaults["seed"]), step=1)
    with param_cols[3]:
        sequence_length = st.number_input(
            "Sequence length",
            min_value=1,
            value=int(training_defaults["sequence_length"]),
            step=1,
        )
        hidden_size = st.number_input(
            "Hidden size",
            min_value=1,
            value=int(training_defaults["hidden_size"]),
            step=1,
        )

    payload: dict[str, Any] = {}
    if dataset is not None and model is not None:
        store_workflow_context(
            st.session_state,
            merge_workflow_context(
                context_from_dataset_row(dataset),
                context_from_model_row(model),
            ),
        )
        st.session_state[ACTIVE_MODEL_ID_KEY] = model["id"]
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
                **{
                    key: value
                    for key, value in training_defaults.items()
                    if key not in {"task_type"}
                },
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "optimizer": optimizer,
                "learning_rate": float(learning_rate),
                "loss_function": loss_function,
                "sequence_length": int(sequence_length),
                "hidden_size": int(hidden_size),
                "seed": int(seed),
            },
            metadata=model.get("metadata") or {},
        )
    with st.expander("Queued training payload preview", expanded=False):
        st.json(payload)
    queue_disabled = not is_queueable_experiment_context(
        dataset=dataset,
        model=model,
        feature_set=feature_set,
        compatibility_blockers=compatibility_blockers,
        override_compatibility=override_compatibility,
    )
    submitted = st.form_submit_button("Queue training job", disabled=queue_disabled)

if submitted:
    catalog = _catalog()
    experiment_id, queued = create_and_enqueue_training_experiment(
        catalog=catalog,
        name=name,
        payload=payload,
    )
    st.success(
        f"Queued training job #{queued.catalog_job_id} for experiment #{experiment_id}."
    )

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
        width="stretch",
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
        st.dataframe(pd.DataFrame(metric_rows), width="stretch")
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
