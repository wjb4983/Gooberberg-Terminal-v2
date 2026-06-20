"""Minimal training runner with synthetic data and artifact output."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from quant_platform.experiments.artifacts import (
    experiment_artifact_dir,
    write_training_artifacts,
)
from quant_platform.experiments.registry import ExperimentRegistry
from quant_platform.experiments.schemas import ExperimentRunResult
from quant_platform.models.factory import build_model_from_dict
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
    experiment_id = registry.create_experiment(
        training_config.experiment_name,
        status="running",
        parameters=training_config.jsonable(),
        metadata={
            "dataset_name": training_config.dataset_name,
            "dataset_version": training_config.dataset_version,
            "model_name": training_config.model_name,
            "model_type": training_config.model_type.value,
            "task_type": training_config.task_type.value,
            "walk_forward_enabled": training_config.walk_forward.enabled,
            "early_stopping_enabled": training_config.early_stopping.enabled,
        },
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
