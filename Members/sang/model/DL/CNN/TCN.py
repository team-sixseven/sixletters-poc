"""TCNECG with a common forward_features/feature_dim interface."""

from __future__ import annotations

import torch
from torch import nn

from ..activations import make_activation as _activation


class _Chomp1d(nn.Module):
    def __init__(self, count: int) -> None:
        super().__init__()
        self.count = count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.count == 0 else x[..., : -self.count]


class _TemporalBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        dilation: int,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            _activation(activation),
            nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            _activation(activation),
        )
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.output_activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.network(x) + self.residual(x))


class TCNECG(nn.Module):
    """Causal temporal convolution network for one ECG window."""

    def __init__(
        self,
        channels: int = 3,
        hidden_dim: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4),
        kernel_size: int = 3,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = channels
        for dilation in dilations:
            blocks.append(
                _TemporalBlock1D(in_channels, hidden_dim, kernel_size, dilation, activation)
            )
            in_channels = hidden_dim
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = hidden_dim
        self.classifier = nn.Linear(self.feature_dim, 1)
        self.output_activation = _activation(activation)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.output_activation(self.features(x))).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
