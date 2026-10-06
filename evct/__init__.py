"""Shared model API, imported lazily so runtime settings can precede torch."""
def __getattr__(name):
    if name in ("build_model", "build_component", "ModularClassifier"):
        from . import model
        return getattr(model, name)
    if name == "load_checkpoint":
        from .checkpoints import load_checkpoint
        return load_checkpoint
    raise AttributeError(name)
