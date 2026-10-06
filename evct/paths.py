"""Resolve paths explicitly relative to a config file, never the working directory."""
import copy
import json
from pathlib import Path

def load_model_config(path):
    path = Path(path).resolve()
    config = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    def resolve(value):
        if value and (value.startswith(("./", "../", "/")) or (path.parent / value).exists()):
            return str((path.parent / value).resolve())
        return value
    for key in ("checkpoint", "image_processor", "tokenizer"):
        if config.get(key):
            config[key] = resolve(config[key])
    for component in ("image_encoder", "text_encoder"):
        spec = config["model"].get(component)
        if spec and spec.get("model_name"):
            spec["model_name"] = resolve(spec["model_name"])
    spec = config.get("feature_image_encoder")
    if spec and spec.get("model_name"):
        spec["model_name"] = resolve(spec["model_name"])
    return config

def portable_model_config(config, destination):
    """Write local paths relative to the exported config, keeping hub IDs intact."""
    import os
    result = copy.deepcopy(config)
    def relative(value):
        return os.path.relpath(value, destination) if value and Path(value).is_absolute() else value
    for key in ("checkpoint", "image_processor", "tokenizer"):
        if result.get(key):
            result[key] = relative(result[key])
    for spec in (result["model"].get("image_encoder"), result["model"].get("text_encoder"),
                 result.get("feature_image_encoder")):
        if spec and spec.get("model_name"):
            spec["model_name"] = relative(spec["model_name"])
    return result
