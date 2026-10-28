"""Dynamic-position Transformer feature encoder for patient segments."""

import math

import torch
from torch import nn

from ..activations import make_activation
from ..models import CNNTST, ECGTransformer


class TransformerSequence(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim=64,
        heads=4,
        layers=2,
        feedforward_dim=128,
        dropout=0.1,
        activation="GELU",
    ):
        super().__init__()
        if embedding_dim < 1 or heads < 1 or embedding_dim % heads:
            raise ValueError("Transformer embedding_dim must be positive and divisible by heads")
        self.feature_dim = embedding_dim
        self.projection = (
            nn.Identity() if input_dim == embedding_dim else nn.Linear(input_dim, embedding_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation=make_activation(activation),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)

    def forward_features(self, features, mask, positions):
        features = self.projection(features)
        frequencies = torch.exp(
            torch.arange(0, self.feature_dim, 2, device=features.device, dtype=features.dtype)
            * (-math.log(10000.0) / self.feature_dim)
        )
        angles = positions[..., None] * frequencies
        positional = torch.zeros_like(features)
        positional[..., 0::2] = torch.sin(angles)
        positional[..., 1::2] = torch.cos(angles[..., : self.feature_dim // 2])
        features = (features + positional).masked_fill(~mask[..., None], 0)
        return self.encoder(features, src_key_padding_mask=~mask).masked_fill(~mask[..., None], 0)


__all__ = ["ECGTransformer", "CNNTST", "TransformerSequence"]
