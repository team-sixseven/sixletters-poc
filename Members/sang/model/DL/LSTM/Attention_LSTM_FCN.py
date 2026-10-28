"""Parallel packed LSTM and masked temporal FCN features, pooled by patient MIL."""

import torch
from torch import nn

from ..activations import make_activation
from ..models import AttentionLSTMFCN
from .BiLSTM import packed_features


class AttentionLSTMFCNSequence(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=32, layers=1, dropout=0.0, activation="GELU", fcn_kernels=(3, 5)
    ):
        super().__init__()
        self.feature_dim = hidden_dim * 2
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        if not fcn_kernels or any(k < 1 or k % 2 == 0 for k in fcn_kernels):
            raise ValueError("fcn_kernels must contain positive odd kernel sizes")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for kernel in fcn_kernels:
            self.convs.append(nn.Conv1d(input_dim, hidden_dim, kernel, padding=kernel // 2))
            self.norms.append(nn.LayerNorm(hidden_dim))
            input_dim = hidden_dim
        self.activation = make_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward_features(self, features, mask, positions):
        recurrent = packed_features(self.lstm, features, mask)
        convolutional = features
        for conv, norm in zip(self.convs, self.norms, strict=True):
            # Per-position LayerNorm does not learn statistics from padded times.
            convolutional = convolutional.masked_fill(~mask[..., None], 0)
            convolutional = conv(convolutional.transpose(1, 2)).transpose(1, 2)
            convolutional = self.dropout(self.activation(norm(convolutional)))
        convolutional = convolutional.masked_fill(~mask[..., None], 0)
        return torch.cat((recurrent, convolutional), dim=-1)


__all__ = ["AttentionLSTMFCN", "AttentionLSTMFCNSequence"]
