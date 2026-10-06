"""Feature modules return [batch, feature_dim]; heads return [batch, labels]."""
import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class DinoImageEncoder(nn.Module):
    """Released DINO pooling: CLS concatenated with mean non-prefix patch tokens."""
    def __init__(self, model_name, dtype="bfloat16", local_files_only=True, frozen=False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only,
            torch_dtype=getattr(torch, dtype))
        self.output_dim = self.backbone.config.hidden_size * 2
        self.prefix_tokens = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        if frozen:
            self.backbone.requires_grad_(False)

    def forward(self, pixel_values):
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state
        return torch.cat([hidden[:, 0], hidden[:, self.prefix_tokens:].mean(1)], dim=1)


class HFTextEncoder(nn.Module):
    """HF text backbone with explicit pooling; checkpoint supplies fine-tuned weights."""
    def __init__(self, model_name, pooling="masked_mean", local_files_only=True,
                 pretrained=False):
        super().__init__()
        if pooling not in ("masked_mean", "cls"):
            raise ValueError("pooling must be masked_mean or cls")
        config = AutoConfig.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = (AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
                         if pretrained else AutoModel.from_config(config))
        self.pooling = pooling
        self.output_dim = config.hidden_size

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


class ImageMLP(nn.Module):
    required_modalities = frozenset({"image"})
    def __init__(self, image_dim, labels, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.image_dim, self.labels = image_dim, labels
        self.head = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, labels))

    def forward(self, *, image_features=None, text_features=None):
        return self.head(image_features.float())


class TextMLP(nn.Module):
    """Optional text-only head. A separately trained matching checkpoint is required."""
    required_modalities = frozenset({"text"})
    def __init__(self, text_dim, labels, hidden_dim=1024, dropout=0.2):
        super().__init__()
        self.text_dim, self.labels = text_dim, labels
        self.head = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden_dim),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, labels))

    def forward(self, *, image_features=None, text_features=None):
        return self.head(text_features.float())


class AttentionFusion(nn.Module):
    required_modalities = frozenset({"image", "text"})
    def __init__(self, image_dim, text_dim, labels, hidden_dim=1024, num_heads=8, dropout=0.2):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.image_dim, self.text_dim, self.labels = image_dim, text_dim, labels
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, labels)

    def forward(self, *, image_features=None, text_features=None):
        sequence = torch.cat([self.text_proj(text_features).unsqueeze(1),
                              self.image_proj(image_features).unsqueeze(1)], dim=1)
        attended, _ = self.attn(sequence, sequence, sequence)
        sequence = self.ln(sequence + attended)
        return self.head(self.drop(sequence.flatten(1)))
