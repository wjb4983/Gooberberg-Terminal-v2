"""Multilayer perceptron model definitions."""

from __future__ import annotations

import torch
from torch import nn

from quant_platform.models.schemas import MLPConfig


class MLP(nn.Module):
    """Feed-forward network for tabular inputs or flattened sequences."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = config.input_dim
        for hidden_dim in config.hidden_dims:
            layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, config.output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Sequence inputs are flattened from ``(batch, sequence, features)`` to
        ``(batch, sequence * features)`` so the model can be reused with either
        tabular or pre-windowed feature tensors.
        """

        if inputs.ndim > 2:
            inputs = inputs.flatten(start_dim=1)
        return self.network(inputs)
