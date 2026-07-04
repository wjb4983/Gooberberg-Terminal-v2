"""Tests for shared experiment queue payload helpers."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest
from apps.api.routes import experiments as experiment_routes

from quant_platform.config.settings import Settings
from quant_platform.data.storage.catalog import MetadataCatalog
from quant_platform.experiments.queueing import (
    ExperimentKind,
    build_training_experiment_payload,
    create_and_enqueue_training_experiment,
)
from quant_platform.jobs.queue import enqueue_training_job
from quant_platform.training.schemas import (
    LossName,
    OptimizerName,
    TargetDefinition,
    TaskType,
)


class _FakeQueue:
    name = "test-queue"

    def __init__(self) -> None:
        self.enqueued: list[dict[str, object]] = []

    def enqueue(self, task_path, catalog_job_id, payload, *, job_id):
        self.enqueued.append(
            {
                "task_path": task_path,
                "catalog_job_id": catalog_job_id,
                "payload": dict(payload),
                "job_id": job_id,
            }
        )


def test_build_training_experiment_payload_shape_and_json_serialization() -> None:
    """Builder should normalize queue inputs into the expected JSON payload shape."""

    payload = build_training_experiment_payload(
        experiment_name=" nightly training ",
        dataset={"id": "7", "name": " prices ", "version": 3},
        feature_set={"id": "9", "features": [" close ", "", "volume"]},
        model={"id": "11", "name": " baseline ", "model_type": " lstm "},
        task_type=TaskType.REGRESSION,
        target=TargetDefinition(name="forward_return", horizon=2),
        split={
            "train_start": date(2024, 1, 1),
            "train_end": date(2024, 1, 31),
            "validation_start": date(2024, 2, 1),
            "validation_end": date(2024, 2, 15),
            "test_start": None,
            "test_end": None,
        },
        training={
            "epochs": 3,
            "batch_size": 32,
            "optimizer": OptimizerName.ADAM,
            "learning_rate": 0.001,
            "loss_function": LossName.MSE,
        },
        metadata={"user": "analyst", "notes": ["smoke"]},
        experiment_id="13",
    )

    assert payload == {
        "experiment_kind": "supervised_training",
        "model_family": "neural_network",
        "experiment_name": "nightly training",
        "dataset_id": 7,
        "dataset_name": "prices",
        "dataset_version": "3",
        "feature_set_id": 9,
        "feature_set": ["close", "volume"],
        "model_id": 11,
        "model_name": "baseline",
        "model_type": "lstm",
        "task_type": "regression",
        "target": {
            "name": "forward_return",
            "horizon": 2,
            "expression": "weighted_feature_sum",
        },
        "split": {
            "train_start": "2024-01-01",
            "train_end": "2024-01-31",
            "validation_start": "2024-02-01",
            "validation_end": "2024-02-15",
            "test_start": None,
            "test_end": None,
        },
        "training": {
            "epochs": 3,
            "batch_size": 32,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss_function": "mse",
        },
        "metadata": {"user": "analyst", "notes": ["smoke"]},
        "experiment_id": 13,
    }
    assert json.loads(json.dumps(payload)) == payload


def test_build_training_experiment_payload_rejects_unsupported_future_kind() -> None:
    """Future experiment kinds should fail validation before they are queued."""

    with pytest.raises(ValueError, match="unsupported experiment kind for queueing"):
        build_training_experiment_payload(
            experiment_name="markov experiment",
            dataset={"id": 7, "name": "prices", "version": "3"},
            feature_set={"id": 9, "features": ["close", "volume"]},
            model={
                "id": 11,
                "name": "markov baseline",
                "model_type": None,
                "metadata": {"model_family": "markov"},
            },
            experiment_kind=ExperimentKind.MARKOV_MODEL,
            task_type=TaskType.REGRESSION,
            target=TargetDefinition(name="forward_return", horizon=2),
            split={
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 1, 31),
                "validation_start": date(2024, 2, 1),
                "validation_end": date(2024, 2, 15),
            },
            training={
                "epochs": 3,
                "batch_size": 32,
                "optimizer": OptimizerName.ADAM,
                "learning_rate": 0.001,
                "loss_function": LossName.MSE,
            },
        )


def test_build_training_experiment_payload_rejects_unsupported_model_family() -> None:
    """Supervised experiments with non-neural model families should not queue."""

    with pytest.raises(ValueError, match="unsupported model family"):
        build_training_experiment_payload(
            experiment_name="custom experiment",
            dataset={"id": 7, "name": "prices", "version": "3"},
            model={
                "id": 11,
                "name": "custom baseline",
                "model_type": "bespoke",
                "metadata": {"model_family": "custom"},
            },
            task_type=TaskType.REGRESSION,
            target=TargetDefinition(),
            split={
                "train_start": date(2024, 1, 1),
                "train_end": date(2024, 1, 31),
                "validation_start": date(2024, 2, 1),
                "validation_end": date(2024, 2, 15),
            },
            training={
                "epochs": 3,
                "batch_size": 32,
                "optimizer": OptimizerName.ADAM,
                "learning_rate": 0.001,
                "loss_function": LossName.MSE,
            },
        )


def test_create_and_enqueue_training_experiment_delegates_to_training_queue(
    tmp_path,
) -> None:
    """Submission helper should create the experiment, then delegate job creation."""

    catalog = MetadataCatalog(tmp_path / "metadata.sqlite")
    catalog.create_all()
    payload = {
        "model_id": 11,
        "dataset_id": 7,
        "feature_set_id": 9,
        "task_type": "regression",
        "target": {"name": "forward_return"},
        "split": {"train_start": "2024-01-01", "train_end": "2024-01-31"},
        "training": {"epochs": 3},
    }
    enqueued_payloads = []

    def fake_enqueue(submitted_payload):
        enqueued_payloads.append(dict(submitted_payload))
        return SimpleNamespace(
            catalog_job_id=42,
            rq_job_id="rq-42",
            queue_name="test-queue",
            job_type="training",
            status="queued",
        )

    experiment_id, queued = create_and_enqueue_training_experiment(
        catalog=catalog,
        name=" nightly training ",
        payload=payload,
        enqueue=fake_enqueue,
    )

    experiments = catalog.list_rows("experiments")
    jobs = catalog.list_rows("jobs")

    assert queued.catalog_job_id == 42
    assert queued.job_type == "training"
    assert queued.status == "queued"
    assert jobs == []
    assert payload["experiment_id"] == experiment_id
    assert enqueued_payloads == [{**payload}]
    assert experiments[0]["id"] == experiment_id
    assert experiments[0]["name"] == "nightly training"
    assert experiments[0]["status"] == "queued"
    assert enqueued_payloads[0]["experiment_id"] == experiment_id


def test_api_and_ui_training_jobs_use_same_job_type(monkeypatch, tmp_path) -> None:
    """API and UI queueing paths should both persist training jobs as training."""

    settings = Settings(catalog_db_path=tmp_path / "metadata.sqlite")
    catalog = MetadataCatalog(settings.catalog_db_path)
    catalog.create_all()
    dataset_id = catalog.insert_row(
        "dataset_definitions",
        {"name": "prices", "version": "1", "schema": {}, "metadata": {}},
    )
    model_id = catalog.insert_row(
        "model_definitions",
        {
            "name": "baseline",
            "version": "1",
            "model_type": "mlp",
            "artifact_uri": None,
            "parameters": {},
            "metadata": {},
        },
    )
    monkeypatch.setattr(experiment_routes, "_catalog", lambda: catalog)

    api_queue = _FakeQueue()
    monkeypatch.setattr(
        experiment_routes,
        "enqueue_training_job",
        lambda payload: enqueue_training_job(
            payload, settings=settings, queue=api_queue
        ),
    )
    api_response = experiment_routes.queue_experiment(
        experiment_routes.ExperimentQueueRequest.model_validate(
            {
                "name": "api training",
                "dataset_id": dataset_id,
                "model_id": model_id,
                "split": {
                    "train_start": "2024-01-01",
                    "train_end": "2024-01-31",
                    "validation_start": "2024-02-01",
                    "validation_end": "2024-02-15",
                },
            }
        )
    )

    ui_payload = build_training_experiment_payload(
        experiment_name="ui training",
        dataset={"id": dataset_id, "name": "prices", "version": "1"},
        model={"id": model_id, "name": "baseline", "model_type": "mlp"},
        task_type=TaskType.REGRESSION,
        target=TargetDefinition(),
        split={
            "train_start": date(2024, 1, 1),
            "train_end": date(2024, 1, 31),
            "validation_start": date(2024, 2, 1),
            "validation_end": date(2024, 2, 15),
        },
        training={"epochs": 1},
    )
    ui_queue = _FakeQueue()
    _, ui_queued = create_and_enqueue_training_experiment(
        catalog=catalog,
        name="ui training",
        payload=ui_payload,
        enqueue=lambda payload: enqueue_training_job(
            payload, settings=settings, queue=ui_queue
        ),
    )

    jobs = catalog.list_rows("jobs")

    assert api_response.job_id != ui_queued.catalog_job_id
    assert {job["job_type"] for job in jobs} == {"training"}
    assert jobs[0]["job_type"] == jobs[1]["job_type"] == "training"
    assert ui_queued.job_type == "training"


def test_build_training_payload_routes_regime_models_to_training_queue() -> None:
    """Regime detector definitions should be routable through training jobs."""

    payload = build_training_experiment_payload(
        experiment_name="regime training",
        dataset={"id": 7, "name": "prices", "version": "1"},
        model={"id": 11, "name": "regime", "model_type": "regime_detector"},
        task_type=TaskType.REGRESSION,
        target=TargetDefinition(),
        split={
            "train_start": date(2024, 1, 1),
            "train_end": date(2024, 1, 2),
            "validation_start": date(2024, 1, 3),
            "validation_end": date(2024, 1, 4),
        },
        training={"epochs": 1},
        metadata={"regime": {"detector_type": "rolling_zscore"}},
    )

    assert payload["model_family"] == "regime"
    assert payload["feature_set"] == []
    assert payload["metadata"] == {"regime": {"detector_type": "rolling_zscore"}}


def test_regime_dataset_model_compatibility_and_override_queueing() -> None:
    """Regime compatibility should gate queueing unless explicitly overridden."""

    from apps.ui.experiment_context import is_queueable_experiment_context
    from apps.ui.experiment_selection import (
        compatibility_messages,
        compatible_model_options,
        is_model_compatible,
    )

    dataset = {
        "id": 7,
        "name": "equity_regime_dataset",
        "version": "1",
        "schema": {"provider": "massive", "data_types": ["daily_bars"]},
        "metadata": {
            "asset_class": "equity",
            "workflow_intent": "learned_regime_switching",
            "regime_labels": True,
        },
    }
    detector = {
        "id": 11,
        "name": "equity regime detector",
        "model_type": "regime_detector",
        "metadata": {
            "workflow_intent": "learned_regime_switching",
            "compatible_asset_classes": ["equity"],
            "regime": {"detector_type": "rolling_zscore"},
        },
    }
    unrelated = {
        "id": 12,
        "name": "crypto forecast",
        "model_type": "mlp",
        "metadata": {
            "workflow_intent": "supervised_forecast",
            "compatible_asset_classes": ["crypto"],
        },
    }

    reasons, warnings = compatibility_messages(dataset, detector)
    assert any("learned regime switching" in reason for reason in reasons)
    assert any("equity" in reason for reason in reasons)
    assert warnings == []
    assert is_model_compatible(dataset, detector)
    assert not is_model_compatible(dataset, unrelated)
    assert compatible_model_options(dataset, [unrelated, detector]) == [detector]
    assert compatible_model_options(
        dataset, [unrelated, detector], show_incompatible=True
    ) == [detector, unrelated]

    blockers = compatibility_messages(dataset, unrelated)[1]
    assert blockers
    assert not is_queueable_experiment_context(
        dataset=dataset,
        model=unrelated,
        feature_set={"id": 1, "features": ["return"]},
        compatibility_blockers=blockers,
    )
    assert is_queueable_experiment_context(
        dataset=dataset,
        model=unrelated,
        feature_set={"id": 1, "features": ["return"]},
        compatibility_blockers=blockers,
        override_compatibility=True,
    )
