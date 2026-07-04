"""Tests for synthetic training runner experiment handling."""

from __future__ import annotations

from datetime import date

from quant_platform.experiments.registry import ExperimentRegistry
from quant_platform.models.schemas import ModelType
from quant_platform.training.runner import default_model_config, run_training
from quant_platform.training.schemas import (
    DateSplitConfig,
    RegimeTrainingConfig,
    TrainingConfig,
)


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
    assert experiments[0]["metadata"]["regime"] == {
        "enabled": False,
        "detector": None,
        "regime_column": "regime",
        "include_regime_feature": True,
        "train_per_regime_models": False,
    }

    metrics = registry.list_metrics()
    assert metrics
    assert {metric["experiment_id"] for metric in metrics} == {experiment_id}
    assert result.manifest.experiment_id == experiment_id


def test_default_training_config_model_config_remains_backward_compatible() -> None:
    """Default regime settings should not change the existing model shape."""

    config = TrainingConfig()

    assert config.regime == RegimeTrainingConfig()
    assert default_model_config(config) == {
        "model_type": "mlp",
        "input_dim": len(config.feature_set) * config.sequence_length,
        "output_dim": 1,
        "dropout": 0.0,
        "hidden_dims": [config.hidden_size],
    }


def test_regime_feature_only_changes_model_config_when_enabled() -> None:
    """Regime feature dimensionality is opt-in via regime.enabled."""

    disabled = TrainingConfig(
        regime=RegimeTrainingConfig(enabled=False, include_regime_feature=True)
    )
    enabled = TrainingConfig(
        regime=RegimeTrainingConfig(enabled=True, include_regime_feature=True)
    )

    assert default_model_config(disabled)["input_dim"] == (
        len(disabled.feature_set) * disabled.sequence_length
    )
    assert default_model_config(enabled)["input_dim"] == (
        (len(enabled.feature_set) + 1) * enabled.sequence_length
    )


def test_run_training_supports_regime_detector_without_feature_set(tmp_path):
    """Regime detector training should infer intended columns from config."""

    registry = ExperimentRegistry(tmp_path / "metadata.sqlite")
    result = run_training(
        TrainingConfig(
            experiment_name="regime-detector-training",
            model_type=ModelType.REGIME_DETECTOR,
            feature_set=[],
            regime=RegimeTrainingConfig(
                enabled=True,
                detector={
                    "detector_type": "rolling_zscore",
                    "feature_column": "return",
                },
            ),
            date_split=DateSplitConfig(
                train_start=date(2024, 1, 1),
                train_end=date(2024, 1, 4),
                validation_start=date(2024, 1, 5),
                validation_end=date(2024, 1, 6),
            ),
            artifact_dir=tmp_path / "artifacts",
        ),
        registry=registry,
    )

    assert result.metrics["rows_labeled"] > 0
    assert result.metrics["regime_count"] >= 1
    assert result.manifest.files["metrics"].endswith("metrics.json")


def test_run_training_supports_regime_switching_without_feature_set(tmp_path):
    """Regime switching training should run intended regime-change tasks."""

    registry = ExperimentRegistry(tmp_path / "metadata.sqlite")
    result = run_training(
        TrainingConfig(
            experiment_name="regime-switching-training",
            model_type=ModelType.REGIME_SWITCHING,
            feature_set=[],
            regime=RegimeTrainingConfig(
                enabled=True,
                detector={
                    "regime_switching": {
                        "switching_type": "switching_linear",
                        "feature_columns": ["return"],
                        "target_column": "target",
                    }
                },
            ),
            date_split=DateSplitConfig(
                train_start=date(2024, 1, 1),
                train_end=date(2024, 1, 4),
                validation_start=date(2024, 1, 5),
                validation_end=date(2024, 1, 6),
            ),
            artifact_dir=tmp_path / "artifacts",
        ),
        registry=registry,
    )

    assert result.metrics["rows_transformed"] > 0
    assert result.metrics["regime_count"] >= 1
