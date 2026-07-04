"""Model definition page for Gooberberg Terminal."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from apps.ui.regime_helpers import (
    model_regime_config,
    model_regime_switching_config,
    parse_csv_columns,
    parse_regime_weights,
)
from apps.ui.workflow_context import (
    REGIME_WORKFLOW_INTENT,
    context_from_dataset_row,
    infer_workflow_intent,
    load_workflow_context,
    merge_workflow_context,
    normalize_workflow_context,
    store_workflow_context,
)

from quant_platform.config import get_settings
from quant_platform.datasets.registry import DatasetRegistry
from quant_platform.models.registry import ModelRegistry
from quant_platform.models.schemas import (
    Activation,
    ModelDefinition,
    ModelType,
    RegimeDetectorType,
    RegimeSwitchingType,
)

MODEL_TEMPLATES: dict[str, dict[str, Any]] = {
    "MLP (tabular/flattened sequence)": {
        "model_type": ModelType.MLP,
        "layer_count": 2,
        "hidden_size": 64,
        "dropout": 0.10,
        "activation": Activation.RELU,
        "sequence_length": 32,
        "input_size": 8,
        "output_size": 1,
    },
    "LSTM (sequence)": {
        "model_type": ModelType.LSTM,
        "layer_count": 2,
        "hidden_size": 64,
        "dropout": 0.10,
        "activation": Activation.TANH,
        "sequence_length": 64,
        "input_size": 8,
        "output_size": 1,
    },
    "GRU (sequence)": {
        "model_type": ModelType.GRU,
        "layer_count": 2,
        "hidden_size": 64,
        "dropout": 0.10,
        "activation": Activation.TANH,
        "sequence_length": 64,
        "input_size": 8,
        "output_size": 1,
    },
    "Temporal CNN": {
        "model_type": ModelType.TEMPORAL_CNN,
        "layer_count": 3,
        "hidden_size": 64,
        "dropout": 0.10,
        "activation": Activation.RELU,
        "sequence_length": 64,
        "input_size": 8,
        "output_size": 1,
    },
    "Transformer Encoder": {
        "model_type": ModelType.TRANSFORMER,
        "layer_count": 2,
        "hidden_size": 64,
        "dropout": 0.10,
        "activation": Activation.GELU,
        "sequence_length": 64,
        "input_size": 8,
        "output_size": 1,
    },
    "Regime Detector": {
        "model_type": ModelType.REGIME_DETECTOR,
        "layer_count": 1,
        "hidden_size": 16,
        "dropout": 0.0,
        "activation": Activation.RELU,
        "sequence_length": 20,
        "input_size": 1,
        "output_size": 1,
    },
    "Regime Switching Allocation": {
        "model_type": ModelType.REGIME_SWITCHING,
        "layer_count": 1,
        "hidden_size": 16,
        "dropout": 0.0,
        "activation": Activation.RELU,
        "sequence_length": 20,
        "input_size": 1,
        "output_size": 1,
    },
}


REAL_DATA_FEATURE_CANDIDATES = ("return", "volatility", "drawdown")
SUPERVISED_TEMPLATE_NAMES = (
    "MLP (tabular/flattened sequence)",
    "LSTM (sequence)",
    "GRU (sequence)",
    "Temporal CNN",
    "Transformer Encoder",
)
REGIME_TEMPLATE_NAMES = ("Regime Detector", "Regime Switching Allocation")


def _dataset_registry() -> DatasetRegistry:
    return DatasetRegistry(get_settings().catalog_db_path)


def _dataset_rows() -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in _dataset_registry().catalog.list_rows("dataset_definitions")
    ]


def _dataset_label(row: dict[str, Any]) -> str:
    return f"#{row['id']} {row['name']} v{row.get('version') or '1'}"


def _metadata_list(metadata: dict[str, Any], *keys: str) -> list[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            items = [item.strip() for item in value.replace("\n", ",").split(",")]
        elif isinstance(value, list | tuple):
            items = [str(item).strip() for item in value]
        else:
            items = []
        if items:
            return [item for item in items if item]
    return []


def _default_feature_columns(dataset_metadata: dict[str, Any]) -> list[str]:
    feature_set_columns = _metadata_list(
        dataset_metadata, "feature_set_columns", "feature_columns", "features"
    )
    candidates = [*feature_set_columns, *REAL_DATA_FEATURE_CANDIDATES]
    seen: set[str] = set()
    return [col for col in candidates if not (col in seen or seen.add(col))] or [
        "return"
    ]


def _default_target_column(dataset_metadata: dict[str, Any]) -> str:
    return (
        _metadata_list(dataset_metadata, "target_columns", "targets", "label_columns")
        or [str(dataset_metadata.get("target_column") or "target")]
    )[0]


def _recommended_template_names(dataset_metadata: dict[str, Any]) -> tuple[str, ...]:
    if dataset_metadata.get("workflow_intent") == REGIME_WORKFLOW_INTENT:
        return REGIME_TEMPLATE_NAMES
    return SUPERVISED_TEMPLATE_NAMES


def _compatibility_metadata(
    *,
    model_type: ModelType,
    workflow_intent: str,
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "workflow_intent": workflow_intent,
        "compatible_asset_classes": _metadata_list(dataset_metadata, "asset_classes")
        or (
            [dataset_metadata["asset_class"]]
            if dataset_metadata.get("asset_class")
            else []
        ),
        "requires_regime_labels": model_type == ModelType.REGIME_SWITCHING,
        "produces_regime_labels": model_type == ModelType.REGIME_DETECTOR,
        "model_family": model_type.value,
    }


def _registry() -> ModelRegistry:
    return ModelRegistry(get_settings().catalog_db_path)


def _definition_rows() -> list[dict[str, Any]]:
    rows = []
    for definition in _registry().list():
        rows.append(
            {
                "name": definition.name,
                "version": definition.version,
                "model_type": definition.model_type.value,
                "layers": definition.layer_count,
                "hidden_size": definition.hidden_size,
                "dropout": definition.dropout,
                "activation": definition.activation.value,
                "sequence_length": definition.sequence_length,
                "input_size": definition.input_size,
                "output_size": definition.output_size,
                "regime_detector": definition.metadata.get("regime"),
                "regime_switching": definition.metadata.get("regime_switching"),
            }
        )
    return rows


st.set_page_config(page_title="Model Definitions", page_icon="🧠", layout="wide")
st.title("Model Definitions")
st.caption(
    "Create reusable neural network definitions and persist them in the "
    "model registry/API catalog."
)

dataset_rows = _dataset_rows()
active_context = load_workflow_context(st.session_state)
active_dataset_index = 0
if dataset_rows and active_context.dataset_id is not None:
    active_dataset_index = next(
        (
            index
            for index, row in enumerate(dataset_rows)
            if int(row["id"]) == active_context.dataset_id
        ),
        0,
    )

st.subheader("Active dataset context")
if dataset_rows:
    selected_dataset_row = st.selectbox(
        "Dataset",
        dataset_rows,
        index=active_dataset_index,
        format_func=_dataset_label,
        help=(
            "Select an existing dataset to drive template recommendations and defaults."
        ),
    )
    dataset_context = context_from_dataset_row(selected_dataset_row)
    active_context = merge_workflow_context(active_context, dataset_context)
    store_workflow_context(st.session_state, active_context)
else:
    selected_dataset_row = None
    st.info(
        "No saved datasets found. Defaults will use any workflow context "
        "already in this session."
    )

dataset_metadata = dict(active_context.dataset_metadata)
recommended_template_names = _recommended_template_names(dataset_metadata)
recommended_templates = [
    name for name in recommended_template_names if name in MODEL_TEMPLATES
]
template_options = recommended_templates + [
    name for name in MODEL_TEMPLATES if name not in recommended_templates
]
default_feature_columns = _default_feature_columns(dataset_metadata)
default_features_raw = ", ".join(default_feature_columns)
default_target_column = _default_target_column(dataset_metadata)
default_input_size = max(1, len(default_feature_columns))
default_sequence_length = int(
    dataset_metadata.get("sequence_length") or dataset_metadata.get("lookback") or 64
)

with st.form("model_definition"):
    st.subheader("Model template selector")
    if recommended_templates:
        st.caption(
            "Recommended for active dataset: " + ", ".join(recommended_templates)
        )
    template_name = st.selectbox("Template", template_options, index=0)
    template = dict(MODEL_TEMPLATES[template_name])
    template["model_type"] = MODEL_TEMPLATES[template_name]["model_type"]
    if active_context.dataset_id is not None:
        template["input_size"] = default_input_size
        template["sequence_length"] = default_sequence_length
        template["output_size"] = 1

    left, right = st.columns(2)
    with left:
        default_name = (
            "baseline_regime_switching"
            if template["model_type"] == ModelType.REGIME_SWITCHING
            else "baseline_regime_detector"
            if template["model_type"] == ModelType.REGIME_DETECTOR
            else "baseline_mlp"
        )
        name = st.text_input("Model definition name", value=default_name)
        version = st.text_input("Version", value="1")
        model_type = template["model_type"]
        st.selectbox(
            "Model type",
            [model_type],
            index=0,
            format_func=lambda value: value.value,
            disabled=True,
            help=(
                "Model type is driven by the selected template to prevent "
                "template/type drift."
            ),
        )
        layer_count = st.number_input(
            "Layer count",
            min_value=1,
            max_value=24,
            value=template["layer_count"],
            step=1,
        )
        hidden_size = st.number_input(
            "Hidden size",
            min_value=1,
            max_value=4096,
            value=template["hidden_size"],
            step=8,
        )
    with right:
        dropout = st.number_input(
            "Dropout",
            min_value=0.0,
            max_value=0.99,
            value=template["dropout"],
            step=0.01,
        )
        activation = st.selectbox(
            "Activation",
            list(Activation),
            index=list(Activation).index(template["activation"]),
            format_func=lambda value: value.value,
        )
        sequence_length = st.number_input(
            "Sequence length",
            min_value=1,
            max_value=10000,
            value=template["sequence_length"],
            step=1,
        )
        input_size = st.number_input(
            "Input size",
            min_value=1,
            max_value=10000,
            value=template["input_size"],
            step=1,
        )
        output_size = st.number_input(
            "Output size",
            min_value=1,
            max_value=10000,
            value=template["output_size"],
            step=1,
            help=f"Default target column: {default_target_column}",
        )

    regime_config = None
    regime_switching_config = None
    show_advanced_regime_metadata = model_type not in {
        ModelType.REGIME_DETECTOR,
        ModelType.REGIME_SWITCHING,
    } and st.checkbox("Show advanced regime metadata", value=False)

    if model_type == ModelType.REGIME_DETECTOR or show_advanced_regime_metadata:
        st.subheader("Regime detection")
        regime_enabled = st.checkbox(
            "Enable regime detection",
            value=model_type == ModelType.REGIME_DETECTOR,
            disabled=model_type == ModelType.REGIME_DETECTOR,
        )
        regime_cols = st.columns(5)
        with regime_cols[0]:
            regime_detector_type = st.selectbox(
                "Detector type",
                list(RegimeDetectorType),
                format_func=lambda value: value.value,
                disabled=not regime_enabled,
            )
        with regime_cols[1]:
            regime_lookback = st.number_input(
                "Regime lookback",
                min_value=2,
                value=max(2, int(sequence_length)),
                step=1,
                disabled=not regime_enabled,
            )
        with regime_cols[2]:
            regime_threshold = st.number_input(
                "Regime threshold", value=0.0, step=0.01, disabled=not regime_enabled
            )
        with regime_cols[3]:
            regime_feature_columns_raw = st.text_input(
                "Regime feature columns",
                value=default_features_raw,
                disabled=not regime_enabled,
            )
        with regime_cols[4]:
            regime_weights_raw = st.text_area(
                "Regime-to-weight mapping",
                value="",
                placeholder="high_risk: 0.25",
                disabled=not regime_enabled,
            )
        regime_config = model_regime_config(
            enabled=regime_enabled,
            detector_type=regime_detector_type,
            lookback=int(regime_lookback),
            threshold=float(regime_threshold),
            feature_columns=parse_csv_columns(regime_feature_columns_raw),
            regime_weights=parse_regime_weights(regime_weights_raw),
        )

    if model_type == ModelType.REGIME_SWITCHING or show_advanced_regime_metadata:
        st.subheader("Regime switching")
        switching_enabled = st.checkbox(
            "Enable regime switching",
            value=model_type == ModelType.REGIME_SWITCHING,
            disabled=model_type == ModelType.REGIME_SWITCHING,
        )
        switch_cols = st.columns(5)
        with switch_cols[0]:
            switching_type = st.selectbox(
                "Switching type",
                list(RegimeSwitchingType),
                format_func=lambda value: value.value,
                disabled=not switching_enabled,
            )
        with switch_cols[1]:
            switching_regime_column = st.text_input(
                "Switching regime column",
                value="regime",
                disabled=not switching_enabled,
            )
        with switch_cols[2]:
            switching_signal_column = st.text_input(
                "Switching signal column",
                value="signal",
                disabled=not switching_enabled,
            )
        with switch_cols[3]:
            switching_target_weight_column = st.text_input(
                "Switching target weight column",
                value="target_weight",
                disabled=not switching_enabled,
            )
        with switch_cols[4]:
            switching_features_raw = st.text_input(
                "Switching feature columns",
                value=default_features_raw,
                disabled=not switching_enabled,
            )
        switching_weights_raw = st.text_area(
            "Switching regime-to-weight/leverage mapping",
            value="high_risk: 0.25",
            disabled=not switching_enabled,
        )
        regime_switching_config = model_regime_switching_config(
            enabled=switching_enabled,
            switching_type=switching_type,
            regime_column=switching_regime_column.strip(),
            signal_column=switching_signal_column.strip(),
            target_weight_column=switching_target_weight_column.strip(),
            feature_columns=parse_csv_columns(switching_features_raw),
            regime_weights=parse_regime_weights(switching_weights_raw),
        )

    metadata = {"template": template_name, "target_column": default_target_column}
    if active_context.dataset_id is not None:
        metadata["dataset_id"] = active_context.dataset_id
    if regime_config is not None:
        metadata["regime"] = regime_config
    if regime_switching_config is not None:
        metadata["regime_switching"] = regime_switching_config
    workflow_intent = dataset_metadata.get("workflow_intent") or infer_workflow_intent(
        model_type=model_type.value, model_metadata=metadata
    )
    metadata.update(
        _compatibility_metadata(
            model_type=model_type,
            workflow_intent=workflow_intent,
            dataset_metadata=dataset_metadata,
        )
    )
    store_workflow_context(
        st.session_state,
        normalize_workflow_context(
            model_type=model_type.value,
            model_metadata=metadata,
            workflow_intent=workflow_intent,
        ),
    )

    definition = ModelDefinition(
        name=name.strip(),
        version=version.strip(),
        model_type=model_type,
        layer_count=int(layer_count),
        hidden_size=int(hidden_size),
        dropout=float(dropout),
        activation=activation,
        sequence_length=int(sequence_length),
        input_size=int(input_size),
        output_size=int(output_size),
        metadata=metadata,
    )
    st.subheader("Config preview")
    st.json(definition.to_parameters())
    save_clicked = st.form_submit_button("Save model definition")

if save_clicked:
    model_id = _registry().register(definition)
    st.session_state["workflow_active_model_id"] = model_id
    st.success(f"Saved model definition #{model_id}.")

st.subheader("Model definition table")
rows = _definition_rows()
if rows:
    st.dataframe(pd.DataFrame(rows), width="stretch")
else:
    st.info("No model definitions have been saved yet.")
