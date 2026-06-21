"""Tests for synthetic training runner experiment handling."""

from __future__ import annotations

from datetime import date

from quant_platform.experiments.registry import ExperimentRegistry
from quant_platform.training.runner import run_training
from quant_platform.training.schemas import DateSplitConfig, TrainingConfig


def test_run_training_uses_existing_experiment_id(tmp_path):
    """Training should update, not duplicate, a provided experiment row."""

    registry = ExperimentRegistry(tmp_path / "metadata.sqlite")
    experiment_id = registry.create_experiment(
        "existing-training-run",
        status="created",
        parameters={"stale": True},
        metadata={"stale": True},
    )

    result = run_training(
        TrainingConfig(
            experiment_id=experiment_id,
            experiment_name="existing-training-run",
            date_split=DateSplitConfig(
                train_start=date(2024, 1, 1),
                train_end=date(2024, 1, 1),
                validation_start=date(2024, 1, 2),
                validation_end=date(2024, 1, 2),
            ),
            batch_size=1,
            epochs=1,
            sequence_length=2,
            hidden_size=4,
            synthetic_rows_per_day=1,
            artifact_dir=tmp_path / "artifacts",
        ),
        registry=registry,
    )

    experiments = registry.list_experiments()
    assert result.experiment_id == experiment_id
    assert len(experiments) == 1
    assert experiments[0]["id"] == experiment_id
    assert experiments[0]["status"] == "running"
    assert experiments[0]["parameters"]["experiment_id"] == experiment_id
    assert experiments[0]["metadata"]["dataset_name"] == "synthetic_prices"

    metrics = registry.list_metrics()
    assert metrics
    assert {metric["experiment_id"] for metric in metrics} == {experiment_id}
    assert result.manifest.experiment_id == experiment_id
