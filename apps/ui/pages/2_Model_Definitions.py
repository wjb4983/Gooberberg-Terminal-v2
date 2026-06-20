"""Model definition page for Gooberberg Terminal."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from quant_platform.config import get_settings
from quant_platform.models.registry import ModelRegistry
from quant_platform.models.schemas import Activation, ModelDefinition, ModelType

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
        name = st.text_input("Model definition name", value="baseline_mlp")
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
        metadata={"template": template_name},
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
