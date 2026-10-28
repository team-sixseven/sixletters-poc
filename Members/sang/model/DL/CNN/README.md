# Shared-lead CNN feature extractors

The actual ResNet, InceptionTime and TCN definitions are in `resnet.py`, `inception_time.py`, and `TCN.py`. Each exposes `forward_features()` and a structural `feature_dim` property. `model_set.yaml` contains flat experiment hyperparameters only; the Python registry in `composed.py` selects the class.

The patient pipeline constructs one shared one-lead encoder. All measured leads use the same weights, and measured-lead features are averaged per segment. Absent leads and invalid windows never enter the CNN or its BatchNorm. It does not rely on lead names or ordering, discard the third lead, or instantiate layers when lead count changes. Historical standalone CNN classification exports remain available through `DL/models.py`.

`model_set.yaml` exposes activation for each model and modules for InceptionTime only. The removed settings keep the former YAML values as constructor defaults: ResNet width 32 / blocks (2, 2); InceptionTime filters 16 / kernels (9, 19, 39); TCN hidden_dim 32 / dilations (1, 2, 4) / kernel_size 3. The constructor arguments remain accepted when restoring existing snapshots that explicitly stored them.
