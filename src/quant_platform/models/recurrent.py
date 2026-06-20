"""Reusable recurrent sequence model definitions."""

from __future__ import annotations

import torch
from torch import nn

from quant_platform.models.schemas import ModelType, RecurrentConfig


class RecurrentModel(nn.Module):
    """LSTM or GRU sequence encoder with a linear prediction head."""

    def __init__(self, config: RecurrentConfig) -> None:
        super().__init__()
        recurrent_cls: type[nn.LSTM] | type[nn.GRU]
        if config.model_type == ModelType.LSTM:
            recurrent_cls = nn.LSTM
        elif config.model_type == ModelType.GRU:
            recurrent_cls = nn.GRU
        else:
            msg = f"unsupported recurrent model type: {config.model_type}"
            raise ValueError(msg)

        self.recurrent = recurrent_cls(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )
        directions = 2 if config.bidirectional else 1
        self.head = nn.Linear(config.hidden_dim * directions, config.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence and predict from the final time step."""

        outputs, _ = self.recurrent(inputs)
        return self.head(outputs[:, -1, :])
