"""Unit tests for reusable Streamlit workflow context helpers."""

from __future__ import annotations

from apps.ui.workflow_context import (
    ACTIVE_DATASET_ID_KEY,
    ACTIVE_MODEL_ID_KEY,
    REGIME_WORKFLOW_INTENT,
    context_from_dataset_row,
    context_from_model_row,
    is_real_market_data_dataset,
    is_regime_dataset_context,
    is_regime_switching_model,
    load_workflow_context,
    merge_workflow_context,
    normalize_workflow_context,
    requires_sequence_features,
    store_workflow_context,
)


def test_workflow_context_normalizes_catalog_rows() -> None:
    dataset_context = context_from_dataset_row(
        {
            "id": 42,
            "schema": {"provider": "Schwab", "data_types": ["Daily_Bars"]},
            "metadata": {"asset_class": "Equity"},
        }
    )
    model_context = context_from_model_row(
        {
            "model_type": "regime_switching",
            "metadata": {"regime_switching": {"enabled": True}},
        }
    )

    context = merge_workflow_context(dataset_context, model_context)

    assert context.asset_class == "equity"
    assert context.data_types == ("daily_bars",)
    assert context.provider == "schwab"
    assert context.dataset_id == 42
    assert context.model_type == "regime_switching"
    assert context.workflow_intent == REGIME_WORKFLOW_INTENT
    assert is_regime_dataset_context(context)
    assert is_regime_switching_model(context)
    assert is_real_market_data_dataset(context)
    assert requires_sequence_features(context)


def test_workflow_context_persists_stable_session_keys() -> None:
    session_state: dict[str, object] = {ACTIVE_MODEL_ID_KEY: 7}
    context = normalize_workflow_context(
        asset_class="equity",
        data_types=["daily_bars"],
        provider="mock",
        dataset_id=3,
        model_type="mlp",
        workflow_intent="supervised_forecast",
    )

    store_workflow_context(session_state, context)
    loaded = load_workflow_context(session_state)

    assert session_state[ACTIVE_DATASET_ID_KEY] == 3
    assert session_state[ACTIVE_MODEL_ID_KEY] == 7
    assert loaded.dataset_id == 3
    assert loaded.model_type == "mlp"
    assert not is_real_market_data_dataset(loaded)
    assert not requires_sequence_features(loaded)


def test_dataset_context_preserves_regime_workflow_intent() -> None:
    context = context_from_dataset_row(
        {
            "id": 99,
            "schema": {"provider": "massive", "data_types": ["daily_bars"]},
            "metadata": {
                "asset_class": "equity",
                "workflow_intent": "learned_regime_switching",
            },
        }
    )

    assert context.workflow_intent == REGIME_WORKFLOW_INTENT
    assert is_regime_dataset_context(context)
    assert requires_sequence_features(context)
