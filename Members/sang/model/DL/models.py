"""Neural network architectures for patient-level Holter ECG experiments."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..config import load_yaml
from .CNN.inception_time import InceptionTimeECG
from .CNN.resnet import ResNetECG
from .CNN.TCN import TCNECG


class ECGCNN(nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class ECGTransformer(nn.Module):
    def __init__(
        self,
        channels: int = 3,
        patch_size: int = 50,
        embed_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 64,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv1d(channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch(x).transpose(1, 2)
        positions = torch.arange(tokens.size(1), device=x.device, dtype=x.dtype)
        frequencies = torch.exp(
            torch.arange(0, tokens.size(2), 2, device=x.device, dtype=x.dtype)
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / tokens.size(2))
        )
        position_encoding = torch.zeros_like(tokens)
        position_encoding[:, :, 0::2] = torch.sin(positions[:, None] * frequencies)
        position_encoding[:, :, 1::2] = torch.cos(
            positions[:, None] * frequencies[: tokens.size(2) // 2]
        )
        encoded = self.encoder(tokens + position_encoding)
        return self.classifier(encoded.mean(dim=1))


class CNNLSTM(nn.Module):
    """Shared-style ECG segment CNN followed by an LSTM over ordered segments."""

    def __init__(self, channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.sequence = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ordered_segments, leads, samples)
        batch, segments, leads, samples = x.shape
        encoded = self.segment_encoder(x.reshape(batch * segments, leads, samples))
        encoded = encoded.reshape(batch, segments, -1)
        _, (hidden, _) = self.sequence(encoded)
        return self.classifier(hidden[-1])


class AttentionLSTMFCN(nn.Module):
    """CNN-FCN window encoder plus attention over an ordered segment LSTM."""

    def __init__(self, channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, hidden_dim, 5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.sequence = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, segments, leads, samples = x.shape
        encoded = self.segment_encoder(x.reshape(batch * segments, leads, samples))
        encoded = encoded.reshape(batch, segments, -1)
        sequence, _ = self.sequence(encoded)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        pooled_cnn = encoded.mean(dim=1)
        return self.classifier(torch.cat((attended, pooled_cnn), dim=1))


class BiLSTMECG(nn.Module):
    """CNN-encoded ECG windows followed by a bidirectional segment LSTM."""

    def __init__(self, channels: int = 3, hidden_dim: int = 32, layers: int = 1) -> None:
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, hidden_dim, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.sequence = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, segments, leads, samples = x.shape
        encoded = self.segment_encoder(x.reshape(batch * segments, leads, samples))
        sequence, _ = self.sequence(encoded.reshape(batch, segments, -1))
        return self.classifier(sequence.mean(dim=1))


class CNNTST(nn.Module):
    """CNN segment embeddings followed by a TST-style temporal Transformer encoder."""

    def __init__(
        self,
        channels: int = 3,
        hidden_dim: int = 32,
        heads: int = 4,
        layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ordered_segments, leads, samples)
        batch, segments, leads, samples = x.shape
        tokens = self.segment_encoder(x.reshape(batch * segments, leads, samples))
        tokens = tokens.reshape(batch, segments, -1)
        positions = torch.arange(segments, device=x.device, dtype=x.dtype)
        frequencies = torch.exp(
            torch.arange(0, tokens.size(-1), 2, device=x.device, dtype=x.dtype)
            * (-torch.log(torch.tensor(10000.0, device=x.device)) / tokens.size(-1))
        )
        position_encoding = torch.zeros_like(tokens)
        position_encoding[:, :, 0::2] = torch.sin(positions[None, :, None] * frequencies)
        position_encoding[:, :, 1::2] = torch.cos(
            positions[None, :, None] * frequencies[: tokens.size(-1) // 2]
        )
        encoded = self.encoder(tokens + position_encoding)
        return self.classifier(encoded.mean(dim=1))


class SegmentAttentionMIL(nn.Module):
    """Pool unordered segment feature vectors into one patient-level logit."""

    def __init__(self, input_features: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_features, hidden_dim), nn.Tanh())
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, segments, features)
        encoded = self.encoder(x)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class MILTCN(nn.Module):
    """Encode ordered segment features with dilated convolutions, then attention-pool."""

    def __init__(self, input_features: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(input_features, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, ordered_segments, features)
        encoded = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class RawECGMIL(nn.Module):
    """Encode each raw ECG segment independently and attention-pool per patient."""

    def __init__(self, channels: int = 3, hidden_dim: int = 32) -> None:
        super().__init__()
        self.segment_encoder = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, segments, leads, samples)
        batch, segments, leads, samples = x.shape
        encoded = self.segment_encoder(x.reshape(batch * segments, leads, samples))
        encoded = encoded.reshape(batch, segments, -1)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class SegmentMeanLogit(nn.Module):
    """Apply a per-window model and average its logits into one patient logit."""

    def __init__(self, window_model: nn.Module) -> None:
        super().__init__()
        self.window_model = window_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, windows, leads, samples)
        batch, windows, leads, samples = x.shape
        logits = self.window_model(x.reshape(batch * windows, leads, samples))
        return logits.reshape(batch, windows, 1).mean(dim=1)


WINDOW_MODELS = {
    "ResNet": ResNetECG,
    "InceptionTime": InceptionTimeECG,
    "TCN": TCNECG,
    "ECG Transformer": ECGTransformer,
}
PATIENT_MODELS = {
    "Attention LSTM FCN": AttentionLSTMFCN,
    "BiLSTM": BiLSTMECG,
    "CNN+Transformer": CNNTST,
    "Segment Attention MIL": SegmentAttentionMIL,
    "MIL-TCN": MILTCN,
    "Raw ECG MIL": RawECGMIL,
}


ARCHITECTURE_CONFIGS = (
    Path(__file__).parent / "CNN/model_set.yaml",
    Path(__file__).parent / "LSTM/model_set.yaml",
    Path(__file__).parent / "Transformer/transformer.yaml",
)


def load_model_parameters(name: str, config_paths: tuple[Path, ...] = ARCHITECTURE_CONFIGS) -> dict:
    """Find the selected model's constructor settings in its architecture YAML."""
    return load_model_spec(name, config_paths)


def load_model_spec(name: str, config_paths: tuple[Path, ...] = ARCHITECTURE_CONFIGS) -> dict:
    """Find experiment hyperparameters for a model in architecture YAML."""
    if name in {"Segment Attention MIL", "MIL-TCN"}:
        # Legacy segment-statistics models use their constructor defaults only.
        return {}
    definitions = {}
    for path in config_paths:
        config = load_yaml(path)
        definitions.update(config.get("models", {}))
    try:
        return dict(definitions[name])
    except KeyError as error:
        raise ValueError(f"No YAML architecture configuration found for {name}") from error


def build_model(name: str, parameters: dict | None = None) -> nn.Module:
    """Create one patient-level model using its YAML architecture settings."""
    parameters = load_model_parameters(name) if parameters is None else parameters
    if name in WINDOW_MODELS:
        return SegmentMeanLogit(WINDOW_MODELS[name](**parameters))
    try:
        return PATIENT_MODELS[name](**parameters)
    except KeyError as error:
        raise ValueError(f"Unknown Holter model: {name}") from error
