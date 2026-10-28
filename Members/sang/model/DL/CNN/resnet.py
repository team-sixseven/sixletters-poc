"""ResNetECG with a common forward_features/feature_dim interface."""

from __future__ import annotations

import torch
from torch import nn

from ..activations import make_activation as _activation


class _ResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation_name = activation
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ResNetECG(nn.Module):
    """Residual 1D CNN for one three-lead ECG window."""

    def __init__(
        self,
        channels: int = 3,
        width: int = 32,
        blocks: tuple[int, int] = (2, 2),
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            _activation(activation),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(width, width, blocks[0], 1, activation)
        self.layer2 = self._make_layer(width, width * 2, blocks[1], 2, activation)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = width * 2
        self.classifier = nn.Linear(self.feature_dim, 1)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, count: int, stride: int, activation: str
    ) -> nn.Sequential:
        layers = [_ResidualBlock1D(in_channels, out_channels, stride, activation)]
        layers.extend(
            _ResidualBlock1D(out_channels, out_channels, activation=activation)
            for _ in range(count - 1)
        )
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer2(self.layer1(self.stem(x)))
        return self.pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
