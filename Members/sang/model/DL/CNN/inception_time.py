"""InceptionTimeECG with a common forward_features/feature_dim interface."""

from __future__ import annotations

import torch
from torch import nn

from ..activations import make_activation as _activation


class _InceptionModule1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filters: int = 16,
        kernels: tuple[int, ...] = (9, 19, 39),
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        if len(kernels) != 3:
            raise ValueError("InceptionTime requires exactly three convolution kernel sizes")
        bottleneck = max(8, filters)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck, 1, bias=False)
        self.branches = nn.ModuleList(
            nn.Conv1d(bottleneck, filters, kernel, padding=kernel // 2, bias=False)
            for kernel in kernels
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, filters, 1, bias=False),
        )
        self.norm = nn.BatchNorm1d(filters * 4)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.bottleneck(x)
        outputs = [branch(reduced) for branch in self.branches]
        outputs.append(self.pool_branch(x))
        return self.activation(self.norm(torch.cat(outputs, dim=1)))


class InceptionTimeECG(nn.Module):
    """Compact InceptionTime-style network for one ECG window."""

    def __init__(
        self,
        channels: int = 3,
        filters: int = 16,
        modules: int = 3,
        kernels: tuple[int, ...] = (9, 19, 39),
        activation: str = "ReLU",
    ) -> None:
        super().__init__()
        layers = [_InceptionModule1D(channels, filters, kernels, activation)]
        layers.extend(
            _InceptionModule1D(filters * 4, filters, kernels, activation)
            for _ in range(modules - 1)
        )
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = filters * 4
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))
