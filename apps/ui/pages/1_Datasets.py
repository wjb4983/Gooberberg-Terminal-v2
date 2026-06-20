"""Dataset management page for Gooberberg Terminal."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from quant_platform.datasets.console import (
    asset_class_options,
    build_definition,
    checks_passed,
    data_type_options,
    job_status_rows,
    parse_asset_universe,
    preview_coverage,
    provider_options,
    queue_ingestion,
    register_dataset,
    validate_definition_inputs,
)

st.set_page_config(page_title="Datasets", page_icon="🗂️", layout="wide")
st.title("Datasets")
st.caption("Define logical datasets, preview coverage, and queue ingestion jobs.")

with st.form("dataset_definition"):
    st.subheader("Dataset definition")
    left, right = st.columns(2)
    with left:
        name = st.text_input("Dataset name", value="equity_daily_bars")
        version = st.text_input("Version", value="1")
        provider = st.selectbox("Provider", provider_options(), index=0)
        data_types = st.multiselect(
            "Data types",
            data_type_options(),
            default=["daily_bars"],
        )
    with right:
        asset_class = st.selectbox("Asset class", asset_class_options(), index=0)
        resolution = st.text_input("Resolution", value="1d")
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
        value="AAPL, MSFT, SPY",
        help="Enter symbols or selectors separated by commas or new lines.",
    )
    description = st.text_area("Description", value="")
    mirror_config = st.checkbox("Mirror definition to configs/datasets", value=True)

    actions = st.columns(3)
    preview_clicked = actions[0].form_submit_button("Coverage preview")
    save_clicked = actions[1].form_submit_button("Save dataset")
    queue_clicked = actions[2].form_submit_button("Queue ingestion")

symbols = parse_asset_universe(symbols_raw)
checks = validate_definition_inputs(
    name=name,
    asset_universe=symbols,
    data_types=data_types,
    start=start,
    end=end,
    provider=provider,
    asset_class=asset_class,
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
    )

    if save_clicked:
        dataset_id = register_dataset(definition, mirror_config=mirror_config)
        st.success(f"Saved dataset definition #{dataset_id}.")

    if preview_clicked or queue_clicked:
        preview = preview_coverage(definition)
        st.subheader("Missing data plan")
        st.metric("Requested partitions", preview.requested_count)
        st.metric("Missing partitions", preview.missing_count)
        st.dataframe(pd.DataFrame(preview.rows), use_container_width=True)

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
st.dataframe(pd.DataFrame(job_status_rows()), use_container_width=True)
