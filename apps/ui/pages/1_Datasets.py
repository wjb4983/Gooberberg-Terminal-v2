"""Dataset management page for Gooberberg Terminal."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from apps.ui.workflow_context import (
    normalize_workflow_context,
    store_workflow_context,
)

from quant_platform.datasets.console import (
    asset_class_options,
    build_definition,
    checks_passed,
    data_type_options,
    data_type_options_for_asset_class,
    default_resolution_for_context,
    default_symbols_for_context,
    job_status_rows,
    parse_asset_universe,
    preview_coverage,
    provider_options,
    provider_options_for_asset_class,
    queue_ingestion,
    register_dataset,
    validate_definition_inputs,
    workflow_intent_options,
)

st.set_page_config(page_title="Datasets", page_icon="🗂️", layout="wide")
st.title("Datasets")
st.caption("Define logical datasets, preview coverage, and queue ingestion jobs.")

intent_options = workflow_intent_options()
intent_labels = [label for label, _ in intent_options]
intent_values = {label: value for label, value in intent_options}

with st.form("dataset_definition"):
    st.subheader("Dataset definition")
    context_left, context_right = st.columns(2)
    with context_left:
        selected_intent_label = st.selectbox(
            "Workflow intent",
            intent_labels,
            index=0,
            help=(
                "Choose the downstream workflow so dataset inputs can be "
                "filtered and defaulted appropriately."
            ),
        )
        workflow_intent = intent_values[selected_intent_label]
    with context_right:
        asset_class = st.selectbox("Asset class", asset_class_options(), index=0)

    filtered_provider_options = provider_options_for_asset_class(asset_class)
    filtered_data_type_options = data_type_options_for_asset_class(
        asset_class, workflow_intent
    )
    is_regime_workflow = workflow_intent == "learned_regime_switching"
    default_data_types = (
        ["daily_bars"]
        if "daily_bars" in filtered_data_type_options
        else filtered_data_type_options[:1]
    )
    default_symbols = ", ".join(
        default_symbols_for_context(asset_class, workflow_intent)
    )
    default_resolution = default_resolution_for_context(asset_class, workflow_intent)

    show_advanced_options = False
    if is_regime_workflow:
        with st.expander("Advanced provider and data-type options", expanded=False):
            st.caption(
                "Learned regime switching defaults to real market bars and "
                "broad market proxies. Expand only when you need less-common "
                "data types or provider combinations."
            )
            show_advanced_options = st.checkbox(
                "Show all provider/data-type options",
                value=False,
            )

    available_provider_options = (
        provider_options() if show_advanced_options else filtered_provider_options
    )
    available_data_type_options = (
        data_type_options() if show_advanced_options else filtered_data_type_options
    )
    default_data_types = [
        data_type
        for data_type in default_data_types
        if data_type in available_data_type_options
    ] or available_data_type_options[:1]

    left, right = st.columns(2)
    with left:
        name = st.text_input("Dataset name", value="equity_daily_bars")
        version = st.text_input("Version", value="1")
        provider = st.selectbox("Provider", available_provider_options, index=0)
        data_types = st.multiselect(
            "Data types",
            available_data_type_options,
            default=default_data_types,
            help=(
                "Filtered to bar data for learned regime switching. Use "
                "Advanced to expose reference/news types."
                if is_regime_workflow and not show_advanced_options
                else None
            ),
        )
    with right:
        resolution = st.text_input("Resolution", value=default_resolution)
        default_end = date.today()
        default_start = default_end - timedelta(days=30)
        selected_range = st.date_input(
            "Date range",
            value=(default_start, default_end),
        )
        start, end = (
            selected_range
            if isinstance(selected_range, tuple) and len(selected_range) == 2
            else (default_start, default_end)
        )
    symbols_raw = st.text_area(
        "Asset universe",
        value=default_symbols,
        help="Enter symbols or selectors separated by commas or new lines.",
    )
    description = st.text_area("Description", value="")
    metadata_defaults = {
        "regime_source": "real_market_data" if is_regime_workflow else "",
        "target_style": "regime_labels" if is_regime_workflow else "",
        "preferred_task_type": (
            "regime_classification" if is_regime_workflow else "forecasting"
        ),
        "feature_requirements": (
            "returns, volatility, trend" if is_regime_workflow else ""
        ),
    }
    with st.expander("Compatibility metadata", expanded=is_regime_workflow):
        st.caption(
            "These fields are saved with the dataset so later workflow pages can "
            "infer model/task compatibility."
        )
        meta_left, meta_right = st.columns(2)
        with meta_left:
            regime_source = st.text_input(
                "Regime source", value=metadata_defaults["regime_source"]
            )
            target_style = st.text_input(
                "Target style", value=metadata_defaults["target_style"]
            )
        with meta_right:
            preferred_task_type = st.text_input(
                "Preferred task type",
                value=metadata_defaults["preferred_task_type"],
            )
            feature_requirements_raw = st.text_area(
                "Feature requirements",
                value=metadata_defaults["feature_requirements"],
                help="Comma or newline separated feature hints required downstream.",
            )
    mirror_config = st.checkbox("Mirror definition to configs/datasets", value=True)

    actions = st.columns(3)
    preview_clicked = actions[0].form_submit_button("Coverage preview")
    save_clicked = actions[1].form_submit_button("Save dataset")
    queue_clicked = actions[2].form_submit_button("Queue ingestion")

symbols = parse_asset_universe(symbols_raw)
feature_requirements = [
    requirement.strip()
    for requirement in feature_requirements_raw.replace("\n", ",").split(",")
    if requirement.strip()
]
dataset_metadata = {
    "asset_class": asset_class,
    "workflow_intent": workflow_intent,
    "labeling_intent": selected_intent_label,
}
if regime_source.strip():
    dataset_metadata["regime_source"] = regime_source.strip()
if target_style.strip():
    dataset_metadata["target_style"] = target_style.strip()
if preferred_task_type.strip():
    dataset_metadata["preferred_task_type"] = preferred_task_type.strip()
if feature_requirements:
    dataset_metadata["feature_requirements"] = feature_requirements

store_workflow_context(
    st.session_state,
    normalize_workflow_context(
        asset_class=asset_class,
        data_types=data_types,
        provider=provider,
        workflow_intent=workflow_intent,
        dataset_metadata=dataset_metadata,
    ),
)
checks = validate_definition_inputs(
    name=name,
    asset_universe=symbols,
    data_types=data_types,
    start=start,
    end=end,
    provider=provider,
    asset_class=asset_class,
    provider_options_allowed=available_provider_options,
    data_type_options_allowed=available_data_type_options,
)

st.subheader("Basic validation checks")
for check in checks:
    if check.passed:
        st.success(f"{check.name}: OK")
    else:
        st.error(f"{check.name}: {check.message}")

if checks_passed(checks):
    definition = build_definition(
        name=name,
        version=version,
        provider=provider,
        asset_universe=symbols,
        data_types=data_types,
        start=start,
        end=end,
        asset_class=asset_class,
        resolution=resolution,
        description=description,
        workflow_intent=workflow_intent,
        regime_source=regime_source.strip() or None,
        target_style=target_style.strip() or None,
        preferred_task_type=preferred_task_type.strip() or None,
        feature_requirements=feature_requirements or None,
        labeling_intent=selected_intent_label,
        metadata=dataset_metadata,
    )

    if save_clicked:
        dataset_id = register_dataset(definition, mirror_config=mirror_config)
        store_workflow_context(
            st.session_state,
            normalize_workflow_context(
                asset_class=asset_class,
                data_types=data_types,
                provider=provider,
                workflow_intent=workflow_intent,
                dataset_id=dataset_id,
                dataset_metadata=definition.metadata,
            ),
        )
        st.success(f"Saved dataset definition #{dataset_id}.")

    if preview_clicked or queue_clicked:
        preview = preview_coverage(definition)
        st.subheader("Missing data plan")
        st.metric("Requested partitions", preview.requested_count)
        st.metric("Missing partitions", preview.missing_count)
        st.dataframe(pd.DataFrame(preview.rows), width="stretch")

    if queue_clicked:
        queued = queue_ingestion(definition)
        st.success(
            f"Queued ingestion job #{queued.job_id} for dataset #{queued.dataset_id}."
        )
elif preview_clicked or save_clicked or queue_clicked:
    st.warning(
        "Fix validation errors before previewing, saving, or queueing ingestion."
    )

st.subheader("Job status")
st.dataframe(pd.DataFrame(job_status_rows()), width="stretch")
