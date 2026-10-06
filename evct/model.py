"""Module selection by config, including user-defined 'package.module:Class' types."""
import importlib
from copy import deepcopy
import torch
from torch import nn

BUILTINS = {
    "dino": "evct.modules:DinoImageEncoder",
    "hf_text": "evct.modules:HFTextEncoder",
    "image_mlp": "evct.modules:ImageMLP",
    "text_mlp": "evct.modules:TextMLP",
    "attention_fusion": "evct.modules:AttentionFusion",
}

def build_component(spec, **defaults):
    if spec is None:
        return None
    spec = deepcopy(spec)
    kind = spec.pop("type")
    path = BUILTINS.get(kind, kind)
    if ":" not in path:
        raise ValueError(f"Unknown module type {kind!r}; use a builtin or package.module:Class")
    module, name = path.split(":", 1)
    cls = getattr(importlib.import_module(module), name)
    defaults.update(spec)
    instance = cls(**defaults)
    if not isinstance(instance, nn.Module):
        raise TypeError(f"{path} must implement torch.nn.Module")
    return instance


class ModularClassifier(nn.Module):
    def __init__(self, image_encoder, text_encoder, classifier):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.classifier = classifier
        if not hasattr(classifier, "required_modalities"):
            raise TypeError("classifier must declare required_modalities")
        for modality, encoder in (("image", image_encoder), ("text", text_encoder)):
            if encoder is not None:
                if not hasattr(encoder, "output_dim"):
                    raise TypeError(f"{modality} encoder must declare output_dim")
                wanted = getattr(classifier, f"{modality}_dim", None)
                if wanted is not None and encoder.output_dim != wanted:
                    raise ValueError(f"{modality} dimension mismatch: {encoder.output_dim} != {wanted}")

    def forward(self, *, pixel_values=None, input_ids=None, attention_mask=None,
                image_features=None, text_features=None):
        required = self.classifier.required_modalities
        if pixel_values is not None and image_features is not None:
            raise ValueError("Supply pixels OR precomputed image features, not both")
        if input_ids is not None and text_features is not None:
            raise ValueError("Supply token inputs OR precomputed text features, not both")
        if pixel_values is not None:
            if self.image_encoder is None:
                raise ValueError("No image encoder configured")
            image_features = self.image_encoder(pixel_values)
        if input_ids is not None:
            if self.text_encoder is None or attention_mask is None:
                raise ValueError("Text encoder and attention_mask are required")
            text_features = self.text_encoder(input_ids, attention_mask)
        batches = []
        for modality, features in (("image", image_features), ("text", text_features)):
            if modality in required and features is None:
                raise ValueError(f"Classifier requires {modality} input")
            if modality not in required and features is not None:
                raise ValueError(f"Classifier does not use {modality}; select the correct input mode")
            if features is not None:
                dim = getattr(self.classifier, f"{modality}_dim", None)
                if features.ndim != 2 or (dim is not None and features.shape[1] != dim):
                    raise ValueError(f"Invalid {modality} features; expected [batch, {dim}]")
                batches.append(features.shape[0])
        if len(set(batches)) > 1:
            raise ValueError("Image and text batch sizes must match")
        return self.classifier(image_features=image_features, text_features=text_features)


def build_model(spec):
    image = build_component(spec.get("image_encoder"))
    text = build_component(spec.get("text_encoder"))
    head_spec = deepcopy(spec["classifier"])
    for name, encoder in (("image_dim", image), ("text_dim", text)):
        if encoder is not None:
            head_spec.setdefault(name, encoder.output_dim)
    return ModularClassifier(image, text, build_component(head_spec))
