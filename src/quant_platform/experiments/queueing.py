"""Helpers for building queued experiment training payloads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum, StrEnum
from typing import Any

from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.jobs.queue import QueuedJob, enqueue_training_job
from quant_platform.models.schemas import ModelType

JsonObject = dict[str, Any]


class ExperimentKind(StrEnum):
    """Experiment kinds accepted by queue payload builders."""

    SUPERVISED_TRAINING = "supervised_training"
    MARKOV_MODEL = "markov_model"
    CUSTOM_MODEL = "custom_model"


class ModelFamily(StrEnum):
    """Runtime model-family groups used to route queued experiments."""

    NEURAL_NETWORK = "neural_network"
    MARKOV = "markov"
    CUSTOM = "custom"


_NEURAL_NETWORK_MODEL_TYPES = {model_type.value for model_type in ModelType}


def _jsonable(value: Any) -> Any:
    """Return a JSON-serializable representation for supported model values."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, date | datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(item) for item in value]
    return value


def _mapping(value: Mapping[str, Any] | Any | None, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    raise TypeError(f"{label} must be a mapping or pydantic-style model")


def _required_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required")
    return int(value)


def _required_text(value: Any, label: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _experiment_kind(value: Any) -> ExperimentKind:
    try:
        return (
            value if isinstance(value, ExperimentKind) else ExperimentKind(str(value))
        )
    except ValueError as exc:
        supported = ", ".join(kind.value for kind in ExperimentKind)
        raise ValueError(
            f"unsupported experiment kind: {value!r}; expected one of: {supported}"
        ) from exc


def _metadata_model_family(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("model_family") or metadata.get("family")
    return _optional_text(value)


def _derive_model_family(model_mapping: Mapping[str, Any]) -> str | None:
    model_type = _optional_text(model_mapping.get("model_type"))
    if model_type in _NEURAL_NETWORK_MODEL_TYPES:
        return ModelFamily.NEURAL_NETWORK.value
    metadata = _mapping(model_mapping.get("metadata"), "model.metadata")
    return _metadata_model_family(metadata)


def _validate_supported_routing(kind: ExperimentKind, model_family: str | None) -> None:
    if kind != ExperimentKind.SUPERVISED_TRAINING:
        raise ValueError(
            f"unsupported experiment kind for queueing: {kind.value}; "
            f"only {ExperimentKind.SUPERVISED_TRAINING.value} neural-network "
            "experiments can run today"
        )
    if model_family != ModelFamily.NEURAL_NETWORK.value:
        raise ValueError(
            f"unsupported model family for {kind.value}: {model_family or 'unknown'}; "
            f"only {ModelFamily.NEURAL_NETWORK.value} experiments can run today"
        )


def build_training_experiment_payload(
    *,
    experiment_name: str,
    dataset: Mapping[str, Any],
    model: Mapping[str, Any],
    feature_set: Mapping[str, Any] | None = None,
    dataset_id: int | None = None,
    feature_set_id: int | None = None,
    model_id: int | None = None,
    experiment_kind: ExperimentKind | str = ExperimentKind.SUPERVISED_TRAINING,
    task_type: Any,
    target: Mapping[str, Any] | Any,
    split: Mapping[str, Any] | Any,
    training: Mapping[str, Any] | Any,
    metadata: Mapping[str, Any] | None = None,
    experiment_id: int | None = None,
) -> JsonObject:
    """Build the JSON-safe training payload shared by API and UI queueing flows.

    The builder normalizes identifiers, human-readable names, enum values, dates,
    nested pydantic models, and caller metadata before jobs are persisted or queued.
    """

    dataset_mapping = _mapping(dataset, "dataset")
    model_mapping = _mapping(model, "model")
    feature_mapping = _mapping(feature_set, "feature_set")
    kind = _experiment_kind(experiment_kind)
    model_family = _derive_model_family(model_mapping)
    _validate_supported_routing(kind, model_family)

    resolved_dataset_id = _required_int(
        dataset_id if dataset_id is not None else dataset_mapping.get("id"),
        "dataset_id",
    )
    resolved_model_id = _required_int(
        model_id if model_id is not None else model_mapping.get("id"), "model_id"
    )
    resolved_feature_set_id = (
        int(feature_set_id)
        if feature_set_id is not None
        else (
            int(feature_mapping["id"])
            if feature_mapping and feature_mapping.get("id") is not None
            else None
        )
    )

    payload: JsonObject = {
        "experiment_kind": kind.value,
        "model_family": model_family,
        "experiment_name": _required_text(experiment_name, "experiment_name"),
        "dataset_id": resolved_dataset_id,
        "dataset_name": _required_text(dataset_mapping.get("name"), "dataset_name"),
        "dataset_version": _required_text(
            dataset_mapping.get("version", "1"), "dataset_version"
        ),
        "feature_set_id": resolved_feature_set_id,
        "feature_set": [
            str(feature).strip()
            for feature in (feature_mapping.get("features") or [])
            if str(feature).strip()
        ],
        "model_id": resolved_model_id,
        "model_name": _required_text(model_mapping.get("name"), "model_name"),
        "model_type": _optional_text(model_mapping.get("model_type")),
        "task_type": _required_text(_jsonable(task_type), "task_type"),
        "target": _jsonable(_mapping(target, "target")),
        "split": _jsonable(_mapping(split, "split")),
        "training": _jsonable(_mapping(training, "training")),
        "metadata": _jsonable(dict(metadata or {})),
    }
    if experiment_id is not None:
        payload["experiment_id"] = int(experiment_id)
    return payload


def create_and_enqueue_training_experiment(
    *,
    catalog: MetadataCatalog,
    name: str,
    payload: JsonObject,
    enqueue: Callable[[Mapping[str, Any]], QueuedJob] = enqueue_training_job,
) -> tuple[int, QueuedJob]:
    """Create an experiment row, attach its id to the payload, and enqueue training.

    The metadata catalog stores the experiment first so downstream workers receive
    a payload containing ``experiment_id``. Job persistence and queue status/type
    conventions are delegated to ``enqueue_training_job``.
    """

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
    queued = enqueue(payload)
    return experiment_id, queued
