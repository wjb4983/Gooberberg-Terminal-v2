"""Minimal training runner with synthetic data and artifact output."""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import nn

from quant_platform.experiments.artifacts import (
    experiment_artifact_dir,
    write_training_artifacts,
)
from quant_platform.experiments.registry import ExperimentRegistry
from quant_platform.experiments.schemas import ExperimentRunResult
from quant_platform.models.factory import (
    build_model_from_dict,
    build_regime_detector_from_dict,
)
from quant_platform.models.regime import StateWeightedAllocationModel
from quant_platform.models.regime_switching import (
    MarkovSwitchingModel,
    StateDependentRiskModel,
    SwitchingLinearModel,
    build_regime_switching_model_from_dict,
)
from quant_platform.models.schemas import ModelType
from quant_platform.training.datamodule import SyntheticDataModule
from quant_platform.training.losses import build_loss
from quant_platform.training.metrics import compute_metrics
from quant_platform.training.schemas import OptimizerName, TrainingConfig


def detect_device() -> torch.device:
    """Detect an available GPU device, falling back to CPU without requiring CUDA."""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_model_config(config: TrainingConfig) -> dict[str, Any]:
    """Build a small model config compatible with the selected model type."""

    input_dim = len(config.feature_set)
    if config.regime.enabled and config.regime.include_regime_feature:
        input_dim += 1
    base: dict[str, Any] = {
        "model_type": config.model_type.value,
        "input_dim": input_dim,
        "output_dim": 1,
        "dropout": 0.0,
    }
    if config.model_type == ModelType.MLP:
        base["input_dim"] = input_dim * config.sequence_length
        base["hidden_dims"] = [config.hidden_size]
    elif config.model_type in {ModelType.LSTM, ModelType.GRU}:
        base["hidden_dim"] = config.hidden_size
        base["num_layers"] = 1
        base["bidirectional"] = False
    elif config.model_type == ModelType.TEMPORAL_CNN:
        base["channels"] = [config.hidden_size]
        base["kernel_size"] = 3
    elif config.model_type == ModelType.TRANSFORMER:
        base["d_model"] = config.hidden_size
        base["nhead"] = 4 if config.hidden_size % 4 == 0 else 1
        base["num_layers"] = 1
        base["dim_feedforward"] = config.hidden_size * 2
    return base


def build_optimizer(
    name: OptimizerName,
    parameters: Any,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build an optimizer for model parameters."""

    if name == OptimizerName.ADAM:
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == OptimizerName.SGD:
        return torch.optim.SGD(parameters, lr=learning_rate)
    raise ValueError(f"unsupported optimizer: {name}")


def run_training(
    config: TrainingConfig | None = None,
    *,
    registry: ExperimentRegistry | None = None,
) -> ExperimentRunResult:
    """Run minimal synthetic training and write artifacts."""

    training_config = config or TrainingConfig()
    torch.manual_seed(training_config.seed)
    device = detect_device()

    registry = registry or ExperimentRegistry()
    experiment_metadata = {
        "dataset_name": training_config.dataset_name,
        "dataset_version": training_config.dataset_version,
        "model_name": training_config.model_name,
        "model_type": training_config.model_type.value,
        "task_type": training_config.task_type.value,
        "walk_forward_enabled": training_config.walk_forward.enabled,
        "early_stopping_enabled": training_config.early_stopping.enabled,
        "regime": training_config.regime.model_dump(mode="json"),
    }
    if training_config.experiment_id is None:
        experiment_id = registry.create_experiment(
            training_config.experiment_name,
            status="running",
            parameters=training_config.jsonable(),
            metadata=experiment_metadata,
        )
    else:
        experiment_id = training_config.experiment_id
        registry.start_experiment(
            experiment_id,
            parameters=training_config.jsonable(),
            metadata=experiment_metadata,
        )

    if training_config.model_type == ModelType.REGIME_DETECTOR:
        return _run_regime_detector_training(
            training_config, registry, experiment_id, experiment_metadata, device
        )
    if training_config.model_type == ModelType.REGIME_SWITCHING:
        return _run_regime_switching_training(
            training_config, registry, experiment_id, experiment_metadata, device
        )

    datamodule = SyntheticDataModule(training_config)
    loaders = datamodule.dataloaders()
    model = build_model_from_dict(default_model_config(training_config)).to(device)
    loss_fn = build_loss(training_config.loss_function)
    optimizer = build_optimizer(
        training_config.optimizer,
        model.parameters(),
        training_config.learning_rate,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, training_config.epochs + 1):
        train_loss = _train_one_epoch(model, loaders.train, loss_fn, optimizer, device)
        validation_loss, validation_metrics = _evaluate(
            model,
            loaders.validation,
            loss_fn,
            device,
            training_config,
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        registry.log_metric(experiment_id, "train_loss", train_loss, step=epoch)
        registry.log_metric(
            experiment_id, "validation_loss", validation_loss, step=epoch
        )
        for metric_name, metric_value in validation_metrics.items():
            registry.log_metric(
                experiment_id, f"validation_{metric_name}", metric_value, step=epoch
            )

    final_metrics = {
        key: float(value) for key, value in history[-1].items() if key != "epoch"
    }
    artifact_dir = experiment_artifact_dir(
        training_config.artifact_dir,
        training_config.experiment_name,
        experiment_id,
    )
    manifest = write_training_artifacts(
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        model=model,
        config=training_config.jsonable(),
        metrics=final_metrics,
        history=history,
        metadata={"device": str(device)},
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        artifact_dir=artifact_dir,
        device=str(device),
        metrics=final_metrics,
        history=history,
        manifest=manifest,
    )


def _synthetic_regime_frame(config: TrainingConfig) -> pd.DataFrame:
    """Build deterministic tabular data for regime detector/switching training."""

    row_count = max(
        12,
        (config.date_split.train_end - config.date_split.train_start).days
        * config.synthetic_rows_per_day,
    )
    generator = torch.Generator().manual_seed(config.seed)
    base = torch.randn(row_count, max(len(config.feature_set), 1), generator=generator)
    columns = list(config.feature_set) or ["return"]
    frame = pd.DataFrame(base[:, : len(columns)].numpy(), columns=columns)
    if "return" not in frame.columns:
        frame["return"] = frame.iloc[:, 0].astype(float)
    frame["close"] = 100.0 + frame["return"].cumsum()
    frame["volume"] = 1_000_000.0 + frame["return"].abs() * 10_000.0
    frame["benchmark_return"] = frame["return"].rolling(3, min_periods=1).mean()
    frame["signal"] = frame["return"].clip(-1.0, 1.0)
    frame["target_weight"] = frame["signal"]
    frame["volatility"] = (
        frame["return"].rolling(5, min_periods=1).std().fillna(0.1).abs() + 0.01
    )
    frame["regime"] = (
        (frame["return"].abs() > frame["return"].abs().median()).astype(int).astype(str)
    )
    frame["target"] = frame["return"].shift(-1).fillna(0.0)
    return frame


def _run_regime_detector_training(
    training_config: TrainingConfig,
    registry: ExperimentRegistry,
    experiment_id: int,
    experiment_metadata: dict[str, Any],
    device: torch.device,
) -> ExperimentRunResult:
    detector_config = training_config.regime.detector or {
        "detector_type": "rolling_zscore"
    }
    detector = build_regime_detector_from_dict(dict(detector_config))
    frame = _synthetic_regime_frame(training_config)
    detector.fit(frame)
    regimes = detector.predict(frame)
    changes = regimes.astype(str).ne(regimes.astype(str).shift()).sum() - 1
    metrics = {
        "regime_count": float(regimes.nunique()),
        "regime_changes": float(max(int(changes), 0)),
        "rows_labeled": float(len(regimes)),
    }
    history = [{"epoch": 1, **metrics}]
    for name, value in metrics.items():
        registry.log_metric(experiment_id, name, value, step=1)
    artifact_dir = experiment_artifact_dir(
        training_config.artifact_dir, training_config.experiment_name, experiment_id
    )
    manifest = write_training_artifacts(
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        model=nn.Identity(),
        config=training_config.jsonable(),
        metrics=metrics,
        history=history,
        metadata={
            **experiment_metadata,
            "device": str(device),
            "regime_labels": regimes.astype(str).value_counts().to_dict(),
        },
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        artifact_dir=artifact_dir,
        device=str(device),
        metrics=metrics,
        history=history,
        manifest=manifest,
    )


def _run_regime_switching_training(
    training_config: TrainingConfig,
    registry: ExperimentRegistry,
    experiment_id: int,
    experiment_metadata: dict[str, Any],
    device: torch.device,
) -> ExperimentRunResult:
    switching_config = (training_config.regime.detector or {}).get(
        "regime_switching", {"switching_type": "state_weighted_allocation"}
    )
    model = build_regime_switching_model_from_dict(dict(switching_config))
    frame = _synthetic_regime_frame(training_config)
    metric_name = "rows_transformed"
    if isinstance(model, SwitchingLinearModel):
        model.fit(frame)
        transformed = model.transform(frame)
        value = float(transformed[model.config.prediction_column].notna().sum())
    elif isinstance(model, MarkovSwitchingModel):
        model.fit(frame)
        probabilities = model.predict_regime_probabilities()
        value = float(probabilities.shape[0])
        metric_name = "regime_probability_rows"
    elif isinstance(model, (StateWeightedAllocationModel, StateDependentRiskModel)):
        transformed = model.transform_signals(frame)
        value = float(len(transformed))
    else:
        raise TypeError(f"unsupported regime-switching model: {type(model).__name__}")
    metrics = {metric_name: value, "regime_count": float(frame["regime"].nunique())}
    history = [{"epoch": 1, **metrics}]
    for name, metric_value in metrics.items():
        registry.log_metric(experiment_id, name, metric_value, step=1)
    artifact_dir = experiment_artifact_dir(
        training_config.artifact_dir, training_config.experiment_name, experiment_id
    )
    manifest = write_training_artifacts(
        artifact_dir=artifact_dir,
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        model=nn.Identity(),
        config=training_config.jsonable(),
        metrics=metrics,
        history=history,
        metadata={
            **experiment_metadata,
            "device": str(device),
            "switching_config": switching_config,
        },
    )
    return ExperimentRunResult(
        experiment_id=experiment_id,
        experiment_name=training_config.experiment_name,
        artifact_dir=artifact_dir,
        device=str(device),
        metrics=metrics,
        history=history,
        manifest=manifest,
    )


def _train_one_epoch(
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_rows += batch_size
    return total_loss / max(total_rows, 1)


def _evaluate(
    model: nn.Module,
    loader: Any,
    loss_fn: nn.Module,
    device: torch.device,
    config: TrainingConfig,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    predictions_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size
            predictions_list.append(predictions.cpu())
            targets_list.append(targets.cpu())
    predictions_tensor = torch.cat(predictions_list)
    targets_tensor = torch.cat(targets_list)
    return total_loss / max(total_rows, 1), compute_metrics(
        config.task_type,
        predictions_tensor,
        targets_tensor,
    )
