# Transformer segment feature encoder

`transformer_enc.py:TransformerSequence` projects the selected CNN feature dimension to the default 64-dimensional embedding, adds positions based on original segment indices, and uses an explicit key-padding mask. It exposes `forward_features` and `feature_dim`; common MIL/MLP input dimensions follow automatically.

`transformer.yaml` contains experiment hyperparameters only. GELU, ReLU, LeakyReLU and SiLU are accepted for the feed-forward activation. Lead count is handled by the shared CNN before Transformer processing; no channels or input/output shape is specified here. Historical standalone exports remain separate from the composed training path.

`transformer.yaml` exposes only `dropout` and `activation`. Structural constructor defaults retain the former YAML values: `embedding_dim=64`, `heads=4`, `layers=2`, `feedforward_dim=128`. These constructor arguments remain accepted for restoring earlier snapshots with explicit structural settings.
