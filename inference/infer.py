"""
전통문양 감성분류 최종 인퍼런스 (이미지 + 텍스트 4개 모델 앙상블, val F1@5=0.8022).

이미지 경로 + description 텍스트를 받아서 top-5 감성 라벨을 예측한다.
필요한 건 전부 이 디렉토리(config.json, weights/*.pt) 안에 있음.

사용법:
  python infer.py --image /path/to/image.jpg --description "이 문양은 ..."
  python infer.py --image /path/to/image.jpg --description "..." --topk 5

배치로 여러 장 하려면 predict() 함수를 직접 import해서 써도 됨:
  from infer import EnsembleModel
  model = EnsembleModel()
  labels, probs = model.predict(image_path, description)
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent


class DinoV3ImageEncoder(nn.Module):
    """DINOv3 이미지 인코더 (frozen). CLS 토큰 + patch 토큰 평균을 concat.
    train_clip_emotion.py의 ClipVisionEncoder(DINO 분기)와 동일 로직 - 이 패키지를
    외부 의존성 없이 완전히 독립 실행 가능하게 만들기 위해 여기 그대로 옮겨옴."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.num_prefix_tokens = 1 + getattr(self.model.config, "num_register_tokens", 0)
        self.out_dim = self.model.config.hidden_size * 2
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        last_hidden_state = self.model(pixel_values=pixel_values).last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, self.num_prefix_tokens:, :]
        return torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=-1)


class ImageTextFusion(nn.Module):
    """train_fusion_final.py와 동일한 구조 (그대로 복사 - 이 디렉토리만으로 인퍼런스 가능하게)."""

    def __init__(self, text_model_name, image_dim, n_labels, hidden_dim=1024, num_heads=8, dropout=0.2):
        super().__init__()
        self.text_backbone = AutoModel.from_pretrained(text_model_name, use_safetensors=True)
        text_dim = self.text_backbone.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, n_labels)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        hid = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hid * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids, attention_mask, image_feat):
        text_pooled = self.encode_text(input_ids, attention_mask)
        txt_tok = self.text_proj(text_pooled).unsqueeze(1)
        img_tok = self.image_proj(image_feat).unsqueeze(1)
        seq = torch.cat([txt_tok, img_tok], dim=1)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.ln(seq + attn_out)
        flat = self.drop(seq.flatten(1))
        return self.head(flat)


class EnsembleModel:
    def __init__(self, config_path: Path = HERE / "config.json", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.vocab = self.config["label_vocab"]

        # 이미지 인코더 (DINOv3, frozen)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config["image_encoder"])
        self.image_encoder = DinoV3ImageEncoder(self.config["image_encoder"]).to(self.device).eval()

        # 4개 텍스트+융합 모델
        self.models = []
        for spec in self.config["models"]:
            tokenizer = AutoTokenizer.from_pretrained(spec["text_model"])
            model = ImageTextFusion(
                spec["text_model"], self.config["image_dim"], len(self.vocab),
                hidden_dim=self.config["hidden_dim"], num_heads=self.config["num_heads"],
            )
            state_dict = torch.load(HERE / spec["weight_file"], map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device).eval()
            self.models.append({
                "tag": spec["tag"], "model": model, "tokenizer": tokenizer,
                "max_length": spec["max_length"], "weight": spec["ensemble_weight"],
            })
        print(f"로드 완료: 이미지 인코더(DINOv3) + 텍스트 모델 {len(self.models)}개, device={self.device}")

    @torch.no_grad()
    def _encode_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        return self.image_encoder(inputs["pixel_values"])  # [1, image_dim]

    @torch.no_grad()
    def predict(self, image_path: str, description: str, topk: int = 5):
        img_feat = self._encode_image(image_path)
        probs_sum = torch.zeros(len(self.vocab), device=self.device)
        for m in self.models:
            tok = m["tokenizer"]([description], padding="max_length", truncation=True,
                                  max_length=m["max_length"], return_tensors="pt").to(self.device)
            logits = m["model"](tok["input_ids"], tok["attention_mask"], img_feat)
            probs_sum += m["weight"] * torch.sigmoid(logits)[0]

        top = probs_sum.topk(topk)
        labels = [self.vocab[i] for i in top.indices.tolist()]
        scores = top.values.tolist()
        return labels, dict(zip(labels, scores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--description", required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model = EnsembleModel(device=args.device)
    labels, scores = model.predict(args.image, args.description, args.topk)
    print(f"\nTop-{args.topk} 예측:")
    for label in labels:
        print(f"  {label}: {scores[label]:.4f}")


if __name__ == "__main__":
    main()
