"""Shared workflow-selection context for Streamlit pages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

WORKFLOW_CONTEXT_KEY = "workflow_context"
ACTIVE_DATASET_ID_KEY = "workflow_active_dataset_id"
ACTIVE_MODEL_ID_KEY = "workflow_active_model_id"
ACTIVE_WORKFLOW_INTENT_KEY = "workflow_active_intent"

REGIME_WORKFLOW_INTENT = "learned_regime_switching"
SUPERVISED_FORECAST_INTENT = "supervised_forecast"
ALLOCATION_INTENT = "allocation"

_SEQUENCE_MODEL_TYPES = {
    "lstm",
    "gru",
    "temporal_cnn",
    "transformer",
    "regime_detector",
    "regime_switching",
}
_REGIME_DATA_TYPES = {
    "regime_labels",
    "regime_features",
    "regime_signals",
    "regime_states",
}
_REAL_MARKET_PROVIDERS = {
    "schwab",
    "massive",
    "polygon",
    "alpaca",
    "yahoo",
    "iex",
    "interactive_brokers",
}
_SYNTHETIC_PROVIDERS = {"mock", "synthetic", "fixture", "demo"}


@dataclass(frozen=True)
class WorkflowContext:
    """Normalized cross-page selections for dataset/model workflows."""

    asset_class: str | None = None
    data_types: tuple[str, ...] = ()
    provider: str | None = None
    dataset_id: int | None = None
    dataset_metadata: dict[str, Any] = field(default_factory=dict)
    model_type: str | None = None
    model_metadata: dict[str, Any] = field(default_factory=dict)
    workflow_intent: str | None = None

    def to_session_value(self) -> dict[str, Any]:
        """Return a Streamlit-session-safe representation."""

        payload = asdict(self)
        payload["data_types"] = list(self.data_types)
        return payload


def normalize_workflow_context(
    *,
    asset_class: str | None = None,
    data_types: Sequence[str] | None = None,
    provider: str | None = None,
    dataset_id: int | str | None = None,
    dataset_metadata: Mapping[str, Any] | None = None,
    model_type: str | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    workflow_intent: str | None = None,
) -> WorkflowContext:
    """Normalize raw Streamlit/catalog selections into a reusable context."""

    normalized_data_types = tuple(
        str(data_type).strip().lower()
        for data_type in data_types or ()
        if str(data_type).strip()
    )
    normalized_dataset_id = int(dataset_id) if dataset_id not in {None, ""} else None
    return WorkflowContext(
        asset_class=_normalize_optional(asset_class),
        data_types=normalized_data_types,
        provider=_normalize_optional(provider),
        dataset_id=normalized_dataset_id,
        dataset_metadata=dict(dataset_metadata or {}),
        model_type=_normalize_optional(model_type),
        model_metadata=dict(model_metadata or {}),
        workflow_intent=_normalize_optional(workflow_intent),
    )


def context_from_dataset_row(row: Mapping[str, Any] | None) -> WorkflowContext:
    """Build context fields from a dataset catalog row."""

    if row is None:
        return WorkflowContext()
    schema = dict(row.get("schema") or {})
    metadata = dict(row.get("metadata") or {})
    return normalize_workflow_context(
        asset_class=metadata.get("asset_class") or schema.get("asset_class"),
        data_types=schema.get("data_types") or (),
        provider=schema.get("provider"),
        dataset_id=row.get("id"),
        dataset_metadata=metadata,
        workflow_intent=(
            metadata.get("workflow_intent") or schema.get("workflow_intent")
        ),
    )


def context_from_model_row(row: Mapping[str, Any] | None) -> WorkflowContext:
    """Build context fields from a model catalog row."""

    if row is None:
        return WorkflowContext()
    return normalize_workflow_context(
        model_type=row.get("model_type"),
        model_metadata=row.get("metadata") or {},
        workflow_intent=infer_workflow_intent(
            model_type=row.get("model_type"),
            model_metadata=row.get("metadata") or {},
        ),
    )


def merge_workflow_context(*contexts: WorkflowContext) -> WorkflowContext:
    """Merge partial contexts, preferring later non-empty selections."""

    merged = WorkflowContext()
    for context in contexts:
        merged = normalize_workflow_context(
            asset_class=context.asset_class or merged.asset_class,
            data_types=context.data_types or merged.data_types,
            provider=context.provider or merged.provider,
            dataset_id=(
                context.dataset_id
                if context.dataset_id is not None
                else merged.dataset_id
            ),
            dataset_metadata={**merged.dataset_metadata, **context.dataset_metadata},
            model_type=context.model_type or merged.model_type,
            model_metadata={**merged.model_metadata, **context.model_metadata},
            workflow_intent=context.workflow_intent or merged.workflow_intent,
        )
    return merged


def store_workflow_context(session_state: Any, context: WorkflowContext) -> None:
    """Persist active workflow selections using stable Streamlit session keys."""

    context = merge_workflow_context(load_workflow_context(session_state), context)
    session_state[WORKFLOW_CONTEXT_KEY] = context.to_session_value()
    if context.dataset_id is not None:
        session_state[ACTIVE_DATASET_ID_KEY] = context.dataset_id
    if context.model_type:
        session_state["workflow_active_model_type"] = context.model_type
    if context.workflow_intent:
        session_state[ACTIVE_WORKFLOW_INTENT_KEY] = context.workflow_intent


def load_workflow_context(session_state: Mapping[str, Any]) -> WorkflowContext:
    """Load the last active workflow context from Streamlit session state."""

    value = session_state.get(WORKFLOW_CONTEXT_KEY) or {}
    return normalize_workflow_context(**value)


def infer_workflow_intent(
    *, model_type: str | None, model_metadata: Mapping[str, Any] | None = None
) -> str:
    """Infer a coarse workflow intent from a model definition."""

    normalized_model_type = _normalize_optional(model_type)
    metadata = model_metadata or {}
    if (
        normalized_model_type == "regime_switching"
        or metadata.get("regime_switching")
    ):
        return REGIME_WORKFLOW_INTENT
    if normalized_model_type == "regime_detector" or metadata.get("regime"):
        return REGIME_WORKFLOW_INTENT
    if normalized_model_type in {
        "mlp",
        "lstm",
        "gru",
        "temporal_cnn",
        "transformer",
    }:
        return SUPERVISED_FORECAST_INTENT
    return ALLOCATION_INTENT


def is_regime_dataset_context(context: WorkflowContext) -> bool:
    """Return true when a dataset looks suitable for regime workflows."""

    metadata = context.dataset_metadata
    return bool(
        context.workflow_intent == REGIME_WORKFLOW_INTENT
        or metadata.get("regime")
        or metadata.get("regime_labels")
        or set(context.data_types).intersection(_REGIME_DATA_TYPES)
    )


def is_regime_switching_model(context: WorkflowContext) -> bool:
    """Return true when the active model is a regime-switching model."""

    return bool(
        context.model_type == "regime_switching"
        or context.model_metadata.get("regime_switching")
    )


def is_real_market_data_dataset(context: WorkflowContext) -> bool:
    """Return true for contexts backed by a live/external market-data provider."""

    return bool(
        context.provider
        and context.provider not in _SYNTHETIC_PROVIDERS
        and (
            context.provider in _REAL_MARKET_PROVIDERS
            or context.asset_class is not None
        )
    )


def requires_sequence_features(context: WorkflowContext) -> bool:
    """Return true when the workflow/model usually requires ordered windows."""

    return bool(
        context.model_type in _SEQUENCE_MODEL_TYPES
        or context.workflow_intent == REGIME_WORKFLOW_INTENT
        or is_regime_dataset_context(context)
    )


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None
