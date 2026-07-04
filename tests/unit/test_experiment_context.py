"""Tests for UI experiment context default derivation."""

from __future__ import annotations

from apps.ui.experiment_context import (
    default_target_for_context,
    default_training_for_context,
    is_queueable_experiment_context,
)

from quant_platform.models import ModelType
from quant_platform.training.schemas import LossName, TaskType


def test_supervised_defaults_keep_forward_return_training_shape() -> None:
    model = {"model_type": ModelType.MLP.value, "metadata": {}}

    assert default_target_for_context(model) == {
        "name": "forward_return",
        "horizon": 1,
        "expression": "weighted_feature_sum",
    }
    training = default_training_for_context(model)
    assert training["task_type"] == TaskType.REGRESSION
    assert training["loss_function"] == LossName.MSE
    assert training["sequence_length"] == 8


def test_regime_detector_defaults_to_unsupervised_discovery_from_metadata() -> None:
    model = {
        "model_type": ModelType.REGIME_DETECTOR.value,
        "metadata": {
            "regime": {
                "detector_type": "clustering",
                "regime_column": "market_regime",
                "feature_columns": ["return", "volatility"],
            }
        },
    }

    assert default_target_for_context(model) == {
        "name": "market_regime",
        "horizon": 1,
        "expression": "return,volatility",
    }
    training = default_training_for_context(model)
    assert training["task_type"] == TaskType.REGRESSION
    assert training["regime"]["enabled"] is True
    assert training["regime"]["regime_column"] == "market_regime"


def test_regime_detector_defaults_to_classification_when_labels_required() -> None:
    model = {
        "model_type": ModelType.REGIME_DETECTOR.value,
        "metadata": {
            "requires_regime_labels": True,
            "regime": {"label_column": "known_regime"},
        },
    }

    assert default_target_for_context(model) == {
        "name": "known_regime",
        "horizon": 1,
        "expression": "regime_classification",
    }
    training = default_training_for_context(model)
    assert training["task_type"] == TaskType.BINARY_CLASSIFICATION
    assert training["loss_function"] == LossName.BCE_WITH_LOGITS


def test_regime_switching_defaults_to_allocation_columns() -> None:
    model = {
        "model_type": ModelType.REGIME_SWITCHING.value,
        "metadata": {
            "regime_switching": {
                "switching_type": "state_weighted_allocation",
                "target_weight_column": "desired_weight",
                "signal_column": "alpha_signal",
                "regime_column": "market_state",
            }
        },
    }

    assert default_target_for_context(model) == {
        "name": "desired_weight",
        "horizon": 1,
        "expression": "alpha_signal",
    }
    training = default_training_for_context(model)
    assert training["regime"]["enabled"] is True
    assert training["regime"]["regime_column"] == "market_state"
    assert training["regime"]["detector"]["regime_switching"]["signal_column"] == (
        "alpha_signal"
    )


def test_queueable_context_allows_regime_models_without_feature_set() -> None:
    dataset = {"id": 1}
    regime_model = {"model_type": ModelType.REGIME_SWITCHING.value}
    neural_model = {"model_type": ModelType.MLP.value}

    assert is_queueable_experiment_context(
        dataset=dataset,
        model=regime_model,
        feature_set=None,
    )
    assert not is_queueable_experiment_context(
        dataset=dataset,
        model=neural_model,
        feature_set=None,
    )
    assert not is_queueable_experiment_context(
        dataset=dataset,
        model=regime_model,
        feature_set=None,
        compatibility_blockers=["blocked"],
    )
    assert is_queueable_experiment_context(
        dataset=dataset,
        model=regime_model,
        feature_set=None,
        compatibility_blockers=["blocked"],
        override_compatibility=True,
    )
