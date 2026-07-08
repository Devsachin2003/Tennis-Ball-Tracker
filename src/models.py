"""Prototype sequence classifiers for tennis shot prediction."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


ClassifierBackbone = Literal["lstm", "transformer"]


class TennisShotClassifier(nn.Module):
    """Binary classifier for fused tennis time-series windows.

    Args:
        feature_dimension: Number of scalar features per frame.
        hidden_dimension: Latent dimension used by the LSTM or Transformer.
        backbone: Sequential encoder type: ``"lstm"`` or ``"transformer"``.
        num_layers: Number of LSTM layers or Transformer encoder layers.
        dropout: Dropout probability applied before the final classifier.
        num_attention_heads: Attention heads for the Transformer backbone.

    Input:
        Tensor shaped ``(batch_size, sequence_length, feature_dimension)``.

    Output:
        Tensor shaped ``(batch_size, 1)`` containing probabilities in ``[0, 1]``.
    """

    def __init__(
        self,
        feature_dimension: int,
        hidden_dimension: int = 128,
        backbone: ClassifierBackbone = "lstm",
        num_layers: int = 2,
        dropout: float = 0.3,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if feature_dimension <= 0:
            raise ValueError("feature_dimension must be positive.")
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if backbone not in ("lstm", "transformer"):
            raise ValueError("backbone must be either 'lstm' or 'transformer'.")
        if hidden_dimension % num_attention_heads != 0:
            raise ValueError("hidden_dimension must be divisible by num_attention_heads.")

        self.feature_dimension = feature_dimension
        self.hidden_dimension = hidden_dimension
        self.backbone = backbone

        if backbone == "lstm":
            self.sequence_encoder: nn.Module = nn.LSTM(
                input_size=feature_dimension,
                hidden_size=hidden_dimension,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.input_projection = None
        else:
            self.input_projection = nn.Linear(feature_dimension, hidden_dimension)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dimension,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dimension * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.sequence_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dimension, 1)
        self.activation = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return shot probability for a batch of fused sequences."""

        if inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape (batch_size, sequence_length, feature_dimension)."
            )
        if inputs.shape[-1] != self.feature_dimension:
            raise ValueError(
                f"Expected feature_dimension={self.feature_dimension}, got {inputs.shape[-1]}."
            )

        if self.backbone == "lstm":
            # LSTM matrix ops consume (B, T, F) and emit encoded states (B, T, H).
            encoded_sequence, _ = self.sequence_encoder(inputs)
        else:
            if self.input_projection is None:
                raise RuntimeError("input_projection is required for transformer backbone.")

            # Linear projection matrix multiply maps (B, T, F) -> (B, T, H).
            projected = self.input_projection(inputs)
            # Transformer self-attention mixes temporal context over (B, T, H).
            encoded_sequence = self.sequence_encoder(projected)

        # Final timestep selection produces (B, H) from the encoded sequence (B, T, H).
        final_step = encoded_sequence[:, -1, :]
        regularized = self.dropout(final_step)
        # Classifier matrix multiply maps (B, H) -> logits (B, 1).
        logits = self.classifier(regularized)
        return self.activation(logits)
