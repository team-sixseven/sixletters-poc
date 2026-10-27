"""Composable CNN window encoders and patient-level LSTM/Transformer sequence heads."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .models import TCNECG, InceptionTimeECG, ResNetECG


class PositionalEncoding(nn.Module):
    def __init__(self, sequence_length: int, width: int) -> None:
        super().__init__()
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, width, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / width)
        )
        encoding = torch.zeros(sequence_length, width)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        if width > 1:
            encoding[:, 1::2] = torch.cos(position * frequencies[: width // 2])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.encoding.size(1):
            raise ValueError("Observed ECG sequence is longer than configured positional encoding")
        return x + self.encoding[:, : x.size(1)].to(dtype=x.dtype, device=x.device)


class CNNWindowEncoder(nn.Module):
    """Encode each `(lead, sample)` window into one feature vector."""

    def __init__(self, name: str, parameters: dict[str, Any]) -> None:
        super().__init__()
        factories = {
            "ResNet": ResNetECG,
            "InceptionTime": InceptionTimeECG,
            "TCN": TCNECG,
        }
        try:
            self.model = factories[name](**parameters)
        except KeyError as error:
            raise ValueError(
                f"Unsupported CNN encoder {name!r}; supported: {sorted(factories)}"
            ) from error
        if name == "ResNet":
            self.output_dim = int(parameters["width"]) * 2
        elif name == "InceptionTime":
            self.output_dim = int(parameters["filters"]) * 4
        else:
            self.output_dim = int(parameters["hidden_dim"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)


class CNNLSTMClassifier(nn.Module):
    """CNN each ordered window, then aggregate the window sequence with an LSTM."""

    def __init__(
        self,
        encoder: CNNWindowEncoder,
        *,
        sequence_model: str,
        sequence_parameters: dict[str, Any],
        windows_per_patient: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        width = encoder.output_dim
        hidden = int(sequence_parameters["hidden_dim"])
        layers = int(sequence_parameters.get("layers", 1))
        if sequence_model == "Attention LSTM FCN":
            self.sequence = nn.LSTM(width, hidden, num_layers=layers, batch_first=True)
            self.attention = nn.Linear(hidden, 1)
            self.classifier = nn.Linear(hidden + width, 1)
            self.mode = "attention_lstm_fcn"
        elif sequence_model == "BiLSTM":
            self.sequence = nn.LSTM(
                width,
                hidden,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=float(sequence_parameters.get("dropout", 0.0)) if layers > 1 else 0.0,
            )
            self.classifier = nn.Linear(hidden * 2, 1)
            self.mode = "bilstm"
        else:
            raise ValueError(
                f"Unsupported LSTM sequence_model {sequence_model!r}; "
                "supported: Attention LSTM FCN, BiLSTM"
            )
        self.windows_per_patient = windows_per_patient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != self.windows_per_patient:
            raise ValueError(
                f"Expected (batch,{self.windows_per_patient},lead,sample), got {tuple(x.shape)}"
            )
        batch, windows, leads, samples = x.shape
        encoded = self.encoder(x.reshape(batch * windows, leads, samples))
        encoded = encoded.reshape(batch, windows, -1)
        sequence, _ = self.sequence(encoded)
        if self.mode == "attention_lstm_fcn":
            weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=1)
            attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
            pooled = encoded.mean(dim=1)
            return self.classifier(torch.cat((attended, pooled), dim=1))
        return self.classifier(sequence.mean(dim=1))


class CNNTransformerClassifier(nn.Module):
    """CNN each ordered window, add positions, and encode the patient sequence."""

    def __init__(
        self,
        encoder: CNNWindowEncoder,
        *,
        transformer_parameters: dict[str, Any],
        windows_per_patient: int,
    ) -> None:
        super().__init__()
        width = encoder.output_dim
        configured_width = int(transformer_parameters["embed_dim"])
        self.encoder = encoder
        self.projection = (
            nn.Identity() if width == configured_width else nn.Linear(width, configured_width)
        )
        heads = int(transformer_parameters["heads"])
        if configured_width % heads:
            raise ValueError("Transformer embed_dim must be divisible by heads")
        layer = nn.TransformerEncoderLayer(
            d_model=configured_width,
            nhead=heads,
            dim_feedforward=int(transformer_parameters["feedforward_dim"]),
            dropout=float(transformer_parameters.get("dropout", 0.0)),
            batch_first=True,
        )
        self.position = PositionalEncoding(windows_per_patient, configured_width)
        self.encoder_stack = nn.TransformerEncoder(
            layer,
            num_layers=int(transformer_parameters["layers"]),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(configured_width, 1)
        self.windows_per_patient = windows_per_patient

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != self.windows_per_patient:
            raise ValueError(
                f"Expected (batch,{self.windows_per_patient},lead,sample), got {tuple(x.shape)}"
            )
        batch, windows, leads, samples = x.shape
        features = self.encoder(x.reshape(batch * windows, leads, samples))
        features = self.projection(features.reshape(batch, windows, -1))
        features = self.position(features)
        encoded = self.encoder_stack(features)
        return self.classifier(encoded.mean(dim=1))


def build_composed_model(
    choice: dict[str, Any],
    basic: dict[str, Any],
    *,
    choice_path,
) -> nn.Module:
    """Resolve only selected CNN/head YAML blocks, so unused model config cannot break a run."""
    from ..config import load_yaml, resolve_config_path

    combination = choice.get("combination")
    if combination not in {"CNN+LSTM", "CNN+Transformer"}:
        raise ValueError("DL choice.combination must be CNN+LSTM or CNN+Transformer")
    cnn_config = load_yaml(resolve_config_path(choice["cnn_config"], choice_path))
    try:
        cnn_spec = cnn_config["models"][choice["cnn_model"]]
    except KeyError as error:
        raise ValueError(
            f"Unknown configured CNN {choice.get('cnn_model')!r}; "
            f"available: {sorted(cnn_config.get('models', {}))}"
        ) from error
    sequence_name = choice.get("sequence_model")
    if combination == "CNN+LSTM":
        lstm_config = load_yaml(resolve_config_path(choice["lstm_config"], choice_path))
        try:
            sequence_spec = lstm_config["models"][sequence_name]
        except KeyError as error:
            raise ValueError(
                f"Unknown LSTM sequence model {sequence_name!r}; "
                f"available: {sorted(lstm_config.get('models', {}))}"
            ) from error
    else:
        transformer_config = load_yaml(
            resolve_config_path(choice["transformer_config"], choice_path)
        )
        if sequence_name != "transformer_enc":
            raise ValueError("CNN+Transformer requires sequence_model=transformer_enc")
        try:
            sequence_spec = transformer_config["models"][sequence_name]
        except KeyError as error:
            raise ValueError(
                f"Missing Transformer configuration {sequence_name!r} "
                f"in {choice['transformer_config']}"
            ) from error
    return build_composed_model_from_specs(choice, basic, cnn_spec, sequence_spec)


def build_composed_model_from_specs(
    choice: dict[str, Any],
    basic: dict[str, Any],
    cnn_spec: dict[str, Any],
    sequence_spec: dict[str, Any],
) -> nn.Module:
    """Recreate a model from resolved config snapshots saved in its run directory."""
    combination = choice["combination"]
    encoder = CNNWindowEncoder(choice["cnn_model"], cnn_spec["parameters"])
    windows = int(basic["data"]["ecg"]["windows_per_patient"])
    if combination == "CNN+LSTM":
        name = choice.get("sequence_model")
        return CNNLSTMClassifier(
            encoder,
            sequence_model=name,
            sequence_parameters=sequence_spec["parameters"],
            windows_per_patient=windows,
        )
    name = choice.get("sequence_model")
    if name != "transformer_enc" or combination != "CNN+Transformer":
        raise ValueError("CNN+Transformer requires sequence_model=transformer_enc")
    return CNNTransformerClassifier(
        encoder,
        transformer_parameters=sequence_spec["parameters"],
        windows_per_patient=windows,
    )
