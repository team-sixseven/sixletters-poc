"""Dataset adapters; real-data policies are defined separately from the demo."""

__all__ = ["make_demo_dataset"]


def __getattr__(name: str):
    if name == "make_demo_dataset":
        from .demo import make_demo_dataset

        return make_demo_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
