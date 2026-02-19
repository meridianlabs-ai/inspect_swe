"""Model name utilities."""


def inspect_model(model: str | None) -> str | None:
    """Prefix *model* with ``inspect/`` unless it already is or is ``None``."""
    if model and model != "inspect" and not model.startswith("inspect/"):
        return f"inspect/{model}"
    return model
