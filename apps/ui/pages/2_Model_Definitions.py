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

from quant_platform.config import get_settings
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

with st.form("model_definition"):
    st.subheader("Model template selector")
    template_name = st.selectbox("Template", list(MODEL_TEMPLATES), index=0)
    template = MODEL_TEMPLATES[template_name]

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
        model_type = st.selectbox(
            "Model type",
            list(ModelType),
            index=list(ModelType).index(template["model_type"]),
            format_func=lambda value: value.value,
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
        )

    st.subheader("Regime detection (optional)")
    regime_enabled = st.checkbox(
        "Enable regime detection",
        value=template["model_type"] == ModelType.REGIME_DETECTOR,
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
            value=20,
            step=1,
            disabled=not regime_enabled,
        )
    with regime_cols[2]:
        regime_threshold = st.number_input(
            "Regime threshold", value=0.0, step=0.01, disabled=not regime_enabled
        )
    with regime_cols[3]:
        regime_feature_columns_raw = st.text_input(
            "Regime feature columns", value="return", disabled=not regime_enabled
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
    st.subheader("Regime switching (optional)")
    switching_enabled = st.checkbox(
        "Enable regime switching",
        value=template["model_type"] == ModelType.REGIME_SWITCHING,
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
            "Switching regime column", value="regime", disabled=not switching_enabled
        )
    with switch_cols[2]:
        switching_signal_column = st.text_input(
            "Switching signal column", value="signal", disabled=not switching_enabled
        )
    with switch_cols[3]:
        switching_target_weight_column = st.text_input(
            "Switching target weight column",
            value="target_weight",
            disabled=not switching_enabled,
        )
    with switch_cols[4]:
        switching_features_raw = st.text_input(
            "Switching feature columns", value="return", disabled=not switching_enabled
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

    metadata = {"template": template_name}
    if regime_config is not None:
        metadata["regime"] = regime_config
    if regime_switching_config is not None:
        metadata["regime_switching"] = regime_switching_config

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
    st.success(f"Saved model definition #{model_id}.")

st.subheader("Model definition table")
rows = _definition_rows()
if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
else:
    st.info("No model definitions have been saved yet.")
