"""Tests for shared experiment queue payload helpers."""

from __future__ import annotations

import json
from datetime import date

from quant_platform.experiments.queueing import build_training_experiment_payload
from quant_platform.training.schemas import (
    LossName,
    OptimizerName,
    TargetDefinition,
    TaskType,
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
