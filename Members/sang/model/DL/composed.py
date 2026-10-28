"""Shared per-lead CNN → temporal features → masked patient MIL → MLP."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..config import dl_config_paths, load_yaml
from .activations import make_activation
from .CNN.inception_time import InceptionTimeECG
from .CNN.resnet import ResNetECG
from .CNN.TCN import TCNECG
from .LSTM.Attention_LSTM_FCN import AttentionLSTMFCNSequence
from .LSTM.BiLSTM import BiLSTMSequence
from .Transformer.transformer_enc import TransformerSequence

MODEL_FORMAT_VERSION = 2
LEAD_STRATEGY = "shared_single_lead_cnn_masked_mean"
CNN_MODELS = {"ResNet": ResNetECG, "InceptionTime": InceptionTimeECG, "TCN": TCNECG}
SEQUENCE_MODELS = {
    "BiLSTM": BiLSTMSequence,
    "Attention LSTM FCN": AttentionLSTMFCNSequence,
    "transformer_enc": TransformerSequence,
}


def sequence_combination(choice):
    if choice.get("sequence_model") not in SEQUENCE_MODELS:
        raise ValueError(
            f"Unknown sequence model {choice.get('sequence_model')!r}; choose {sorted(SEQUENCE_MODELS)}"
        )
    return "CNN+Transformer" if choice["sequence_model"] == "transformer_enc" else "CNN+LSTM"


def load_dl_specs(choice, choice_path):
    paths = dl_config_paths(choice_path)
    sequence_combination(choice)
    cnn = load_yaml(paths["cnn"])["models"]
    sequence = load_yaml(
        paths["transformer"] if choice["sequence_model"] == "transformer_enc" else paths["lstm"]
    )["models"]
    try:
        return (
            cnn[choice["cnn_model"]],
            sequence[choice["sequence_model"]],
            load_yaml(paths["training"]),
        )
    except KeyError as error:
        raise ValueError(f"Unknown configured DL model: {choice}") from error


def _hyperparameters(values):
    forbidden = {"implementation", "feature_method", "input", "output", "parameters", "channels"}
    if forbidden.intersection(values):
        raise ValueError(
            "DL model YAML must contain flat experiment hyperparameters, without registry/shape/channel fields"
        )
    return values


def _validate_mil_inputs(features, mask):
    if features.ndim != 3 or mask.shape != features.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(
            "MIL expects (batch, segments, features) and a boolean (batch, segments) mask"
        )
    if not mask.any(dim=1).all():
        raise ValueError("MIL requires at least one valid segment per patient")


class MaskedMeanMIL(nn.Module):
    """Average measured temporal features; padded/excluded segments have zero weight."""

    def forward(self, features, mask):
        _validate_mil_inputs(features, mask)
        return features.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)


class MaskedMaxMIL(nn.Module):
    """Feature-wise maximum over valid segments, including when all values are negative."""

    def forward(self, features, mask):
        _validate_mil_inputs(features, mask)
        return features.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)


class MaskedAttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, activation="GELU", dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features, mask):
        _validate_mil_inputs(features, mask)
        features = features.masked_fill(~mask[..., None], 0)
        scores = (
            self.score(features).squeeze(-1).masked_fill(~mask, torch.finfo(features.dtype).min)
        )
        return (features * torch.softmax(scores, 1)[..., None]).sum(1)


def build_mil(input_dim, configuration):
    settings = dict(configuration)
    pooling = settings.pop("pooling", "attention")
    if pooling == "attention":
        return MaskedAttentionMIL(input_dim, **settings)
    factories = {"mean": MaskedMeanMIL, "max": MaskedMaxMIL}
    if pooling not in factories:
        raise ValueError(f"Unknown MIL pooling {pooling!r}; choose attention, mean, or max")
    if settings:
        raise ValueError(
            f"MIL {pooling} has no trainable pooling hyperparameters. "
            f"Remove attention-only settings from mil: {sorted(settings)}"
        )
    return factories[pooling]()


class CNNWindowEncoder(nn.Module):
    """Pool only measured leads; channel identity/order is not inferred from column position."""

    def __init__(self, name, parameters):
        super().__init__()
        if name not in CNN_MODELS:
            raise ValueError(f"Unknown CNN {name!r}; choose {sorted(CNN_MODELS)}")
        try:
            # One shared network is registered once, regardless of observed lead count.
            self.model = CNN_MODELS[name](channels=1, **_hyperparameters(parameters))
        except TypeError as error:
            raise ValueError(f"Invalid {name} hyperparameters: {error}") from error
        self.output_dim = self.model.feature_dim
        self.model.classifier = nn.Identity()

    def forward_features(self, x, window_mask, lead_mask):
        batch, windows, leads, samples = x.shape
        measured = window_mask[..., None] & lead_mask[:, None, :]
        if not measured.any(-1)[window_mask].all():
            raise ValueError("Every usable segment needs at least one measured lead")
        values = x[measured]
        if not torch.isfinite(values).all():
            raise ValueError("Measured ECG samples contain NaN/Inf")
        # Padded leads and excluded times never enter CNN/BatchNorm.
        vectors = self.model.forward_features(values.reshape(-1, 1, samples))
        rows = torch.nonzero(measured, as_tuple=False)
        flat_windows = rows[:, 0] * windows + rows[:, 1]
        total = vectors.new_zeros(batch * windows, self.output_dim)
        total = total.index_add(0, flat_windows, vectors)
        counts = measured.sum(-1).clamp_min(1).reshape(-1, 1)
        return (total / counts).reshape(batch, windows, self.output_dim)

    def forward(self, x, window_mask, lead_mask):
        return self.forward_features(x, window_mask, lead_mask)


class PatientECGClassifier(nn.Module):
    def __init__(self, encoder, sequence_name, sequence_parameters, training):
        super().__init__()
        self.encoder = encoder
        self.gap_projection = nn.Linear(1, encoder.output_dim, bias=False)
        try:
            self.sequence = SEQUENCE_MODELS[sequence_name](
                encoder.output_dim, **_hyperparameters(sequence_parameters)
            )
        except TypeError as error:
            raise ValueError(f"Invalid {sequence_name} hyperparameters: {error}") from error
        self.mil = build_mil(self.sequence.feature_dim, training.get("mil", {}))
        classifier = training.get("classifier", {})
        hidden = int(classifier.get("hidden_dim", 32))
        self.classifier = nn.Sequential(
            nn.Linear(self.sequence.feature_dim, hidden),
            make_activation(classifier.get("activation", "GELU")),
            nn.Dropout(float(classifier.get("dropout", 0.1))),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, mask=None, positions=None, lead_presence=None):
        if x.ndim != 4 or min(x.shape) < 1:
            raise ValueError(
                f"Expected nonempty (batch, segments, leads, samples); got {tuple(x.shape)}"
            )
        batch, windows, leads, _ = x.shape
        if mask is None:
            mask = torch.ones((batch, windows), dtype=torch.bool, device=x.device)
        if mask.dtype != torch.bool or mask.shape != x.shape[:2] or not mask.any(1).all():
            raise ValueError(
                "window mask must be boolean (batch, segments), with a valid segment per patient"
            )
        if lead_presence is None:
            lead_presence = torch.ones((batch, leads), dtype=torch.bool, device=x.device)
        if (
            lead_presence.shape != (batch, leads)
            or not ((lead_presence == 0) | (lead_presence == 1)).all()
        ):
            raise ValueError("lead mask must be binary (batch, leads), separate from the time mask")
        lead_mask = lead_presence.to(dtype=torch.bool)
        if positions is None:
            positions = torch.arange(windows, device=x.device).expand(batch, -1)
        if positions.shape != mask.shape or not torch.isfinite(positions[mask]).all():
            raise ValueError("Original segment positions must be finite for usable segments")
        encoded = self.encoder.forward_features(x, mask, lead_mask)
        # Compact only usable segments; retain their original positions/gaps.
        lengths = mask.sum(1)
        encoded = pad_sequence([encoded[i, mask[i]] for i in range(batch)], batch_first=True)
        positions = pad_sequence([positions[i, mask[i]] for i in range(batch)], batch_first=True)
        mask = torch.arange(encoded.size(1), device=x.device)[None, :] < lengths[:, None]
        gaps = torch.zeros_like(positions, dtype=encoded.dtype)
        gaps[:, 1:] = (positions[:, 1:] - positions[:, :-1]).to(encoded.dtype)
        if (gaps[mask] < 0).any():
            raise ValueError("Usable ECG segments must remain in chronological order")
        gaps = gaps.masked_fill(~mask, 0)
        encoded = (encoded + self.gap_projection(torch.log1p(gaps)[..., None])).masked_fill(
            ~mask[..., None], 0
        )
        temporal = self.sequence.forward_features(encoded, mask, positions)
        return self.classifier(self.mil(temporal, mask))


def build_composed_model(choice, basic, *, choice_path):
    cnn, sequence, training = load_dl_specs(choice, choice_path)
    return build_composed_model_from_specs(choice, basic, cnn, sequence, training)


def build_composed_model_from_specs(choice, basic, cnn_spec, sequence_spec, training=None):
    """Restore from a run snapshot; input dimensions come from encoder interfaces."""
    sequence_combination(choice)
    return PatientECGClassifier(
        CNNWindowEncoder(choice["cnn_model"], cnn_spec),
        choice["sequence_model"],
        sequence_spec,
        training or {},
    )
