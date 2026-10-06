"""Strict, explicit adapters for released checkpoint layouts; no silent partial loading."""
from pathlib import Path
import torch

def map_state_dict(state, model, layout):
    expected = set(model.state_dict())
    mapped = {}
    for key, value in state.items():
        if layout == "legacy_image":
            if key.startswith("backbone."):
                key = "image_encoder." + key
            elif key.startswith("head."):
                key = "classifier." + key
        elif layout == "legacy_fusion":
            if key.startswith("text_backbone."):
                key = "text_encoder.backbone." + key[len("text_backbone."):]
            else:
                key = "classifier." + key
        elif layout != "modular":
            raise ValueError(f"Unknown checkpoint layout: {layout}")
        # DINOv3 wrappers in transformers 4.x and 5.x differ by one model prefix.
        if key not in expected:
            alternatives = [
                key.replace("backbone.layer.", "backbone.model.layer.", 1),
                key.replace("backbone.model.layer.", "backbone.layer.", 1),
            ]
            key = next((x for x in alternatives if x in expected), key)
        if key in mapped:
            raise ValueError(f"Checkpoint key collision: {key}")
        mapped[key] = value
    return mapped

def load_checkpoint(model, path, layout="modular"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Matching trained checkpoint required: {path}")
    # Released image checkpoints contain training metadata, not just tensors.
    payload = torch.load(path, map_location="cpu", weights_only=(layout != "legacy_image"))
    state = payload.get("state_dict", payload)
    model.load_state_dict(map_state_dict(state, model, layout), strict=True)
    return model
