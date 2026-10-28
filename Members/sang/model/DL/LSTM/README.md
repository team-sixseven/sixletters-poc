# Temporal feature extractors

`BiLSTM.py:BiLSTMSequence` packs valid lengths and returns bidirectional segment features. `Attention_LSTM_FCN.py:AttentionLSTMFCNSequence` uses a packed unidirectional LSTM in parallel with masked temporal convolutions over CNN segment features. Their output features are concatenated per segment before the common patient attention MIL and MLP.

Each class exposes `forward_features`/`feature_dim`; its input dimension is supplied by the selected CNN. Original segment gaps are embedded before temporal processing. FCN LayerNorm is per position and padding is cleared between convolution layers. LSTM gate sigmoid/tanh remains unchanged. Historical standalone classification exports are retained for compatibility and are not used by configured training.

`model_set.yaml` exposes only activation/dropout: Attention LSTM FCN keeps `activation` and `dropout`, while BiLSTM keeps `dropout` because its recurrent sigmoid/tanh gates are fixed. Hidden width 32, one recurrent layer and FCN kernels (3, 5) use constructor defaults. BiLSTM applies output dropout even with one layer; for multiple layers its recurrent inter-layer dropout also applies. Evaluation disables dropout.
