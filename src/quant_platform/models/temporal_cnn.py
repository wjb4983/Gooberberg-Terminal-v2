"""Temporal convolutional model definitions."""

from __future__ import annotations

import torch
from torch import nn

from quant_platform.models.schemas import TemporalCNNConfig


class TemporalCNN(nn.Module):
    """Small causal-style temporal CNN for sequence features."""

    def __init__(self, config: TemporalCNNConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_channels = config.input_dim
        padding = config.kernel_size - 1
        for channels in config.channels:
            layers.append(
                nn.Conv1d(
                    previous_channels,
                    channels,
                    config.kernel_size,
                    padding=padding,
                )
            )
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            previous_channels = channels
        self.network = nn.Sequential(*layers)
        self.head = nn.Linear(previous_channels, config.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass over ``(batch, sequence, features)`` inputs."""

        features = inputs.transpose(1, 2)
        encoded = self.network(features)
        pooled = encoded[..., -1]
        return self.head(pooled)
