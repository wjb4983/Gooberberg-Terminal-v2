"""Transformer encoder model definitions."""

from __future__ import annotations

import torch
from torch import nn

from quant_platform.models.schemas import TransformerConfig


class TransformerEncoderModel(nn.Module):
    """Transformer encoder placeholder with projection and prediction head."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.head = nn.Linear(config.d_model, config.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence and predict from the final token representation."""

        projected = self.input_projection(inputs)
        encoded = self.encoder(projected)
        return self.head(encoded[:, -1, :])
