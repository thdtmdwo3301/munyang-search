"""Image-only DINOv3-L inference with the trained MLP and optional feature cache."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from train_adapter_ddp import Adapter


def load_adapter(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    input_dim = state["net.1.weight"].shape[1]
    hidden = state["net.1.weight"].shape[0]
    labels = state["net.7.weight"].shape[0]
    model = Adapter(input_dim, labels, hidden, dropout=0.0)
    model.load_state_dict(state)
    return model.to(device).eval(), checkpoint


class Predictor:
    def __init__(self, model_path, checkpoint_path, features_path, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.adapter, self.checkpoint = load_adapter(checkpoint_path, self.device)
        payload = np.load(features_path, allow_pickle=True)
        self.bank = payload["features"].astype(np.float32)
        self.bank /= np.clip(np.linalg.norm(self.bank, axis=1, keepdims=True), 1e-8, None)
        self.bank_labels = payload["labels"].astype(np.float32)
        self.vocab = payload["vocab"].astype(str).tolist()

    @torch.inference_mode()
    def encode(self, image_path):
        with Image.open(image_path) as image:
            pixels = self.processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ].to(self.device, dtype=torch.bfloat16)
        hidden = self.encoder(pixel_values=pixels).last_hidden_state
        patch_start = 1 + int(getattr(self.encoder.config, "num_register_tokens", 0))
        feature = torch.cat([hidden[:, 0], hidden[:, patch_start:].mean(dim=1)], dim=1).float()
        feature = torch.nn.functional.normalize(feature, dim=1)
        return feature

    @torch.inference_mode()
    def predict(self, image_path, cache_weight=0.5):
        feature = self.encode(image_path)
        mlp_score = torch.sigmoid(self.adapter(feature))[0].cpu().numpy()
        feature_np = feature[0].cpu().numpy()
        nearest = int(np.argmax(self.bank @ feature_np))
        cache_score = self.bank_labels[nearest]
        score = (1.0 - cache_weight) * mlp_score + cache_weight * cache_score
        top = np.argsort(-score)[:5]
        return [(self.vocab[i], float(score[i])) for i in top], nearest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-path", default="/weights/dinov3l")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--cache-weight", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    predictor = Predictor(args.model_path, args.checkpoint, args.features, args.device)
    predictions, nearest = predictor.predict(args.image, args.cache_weight)
    print(f"nearest_cache_index={nearest}")
    for label, score in predictions:
        print(f"{label}\t{score:.6f}")


if __name__ == "__main__":
    main()

