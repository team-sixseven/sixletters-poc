"""Shared experiment-selectable hidden activations; recurrent gates remain unchanged."""

from torch import nn


def make_activation(name: str) -> nn.Module:
    factories = {"GELU": nn.GELU, "ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "SiLU": nn.SiLU}
    if name not in factories:
        raise ValueError(f"Unsupported activation {name!r}; choose one of {sorted(factories)}")
    return factories[name]()
