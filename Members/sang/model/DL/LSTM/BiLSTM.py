"""Packed bidirectional LSTM over patient segment features."""

from torch import nn

from ..models import BiLSTMECG


def packed_features(lstm, features, mask):
    lengths = mask.sum(1).to(device="cpu")
    packed = nn.utils.rnn.pack_padded_sequence(
        features, lengths, batch_first=True, enforce_sorted=False
    )
    output, _ = lstm(packed)
    return nn.utils.rnn.pad_packed_sequence(
        output, batch_first=True, total_length=features.size(1)
    )[0]


class BiLSTMSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, layers=1, dropout=0.0):
        super().__init__()
        self.feature_dim = 2 * hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        # Output dropout remains effective for the default single recurrent layer.
        self.output_dropout = nn.Dropout(dropout)

    def forward_features(self, features, mask, positions):
        return self.output_dropout(packed_features(self.lstm, features, mask))


__all__ = ["BiLSTMECG", "BiLSTMSequence"]
