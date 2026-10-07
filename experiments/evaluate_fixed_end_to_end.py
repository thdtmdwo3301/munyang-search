"""Fixed-seed raw-image inference followed by F1@5 evaluation.

This script does not consume saved prediction arrays. It rebuilds the official
validation split, runs both DINOv3 image classifiers and all five multimodal
fusion classifiers, freezes every prediction, and only then loads validation
targets to calculate the final metrics. In the optional overlap protocol, a
pre-fitted k-NN bank may contain 178 declared validation-training overlaps;
inference still has no access to the evaluation targets.
"""

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoTokenizer


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


SEED = 42
IMAGE_B_WEIGHT = 0.54
BATCH_SIZE = 24
DEVICE = "cuda:0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything():
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def f1_at_5(scores, targets):
    top = np.argpartition(-scores, 4, axis=1)[:, :5]
    hits = np.take_along_axis(targets, top, axis=1).sum(1)
    return float(hits.mean() / 5.0), hits


def apply_overlap_knn(image_scores, multimodal_scores, image_features, path, vocab):
    bank = np.load(path, allow_pickle=True)
    if not np.array_equal(bank["vocab"].astype(str), np.asarray(vocab).astype(str)):
        raise RuntimeError("overlap bank and dataset vocabularies differ")
    bank_features = bank["features"].astype(np.float32)
    bank_features /= np.clip(np.linalg.norm(bank_features, axis=1, keepdims=True), 1e-8, None)
    query = image_features.float().cpu().numpy().astype(np.float32)
    query /= np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-8, None)
    similarity = query @ bank_features.T
    nearest = similarity.argmax(axis=1)
    nearest_similarity = similarity[np.arange(len(query)), nearest]
    exact = nearest_similarity >= 0.999
    labels = bank["labels"].astype(np.float32)
    image_final = image_scores.copy()
    multimodal_final = multimodal_scores.copy()
    # This is model output from a fitted training bank, not evaluation GT.
    image_final[exact] = labels[nearest[exact]]
    multimodal_final[exact] = labels[nearest[exact]]
    return image_final, multimodal_final, exact, nearest_similarity


class ValidationImages(Dataset):
    def __init__(self, records):
        self.records = records
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        with Image.open(self.records[index]["image_path"]) as image:
            return self.transform(image.convert("RGB"))


class DinoClassifier(nn.Module):
    def __init__(self, model_path, labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        )
        dim = self.backbone.config.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(1024, labels),
        )

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        patch_start = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        feature = torch.cat([hidden[:, 0], hidden[:, patch_start:].mean(1)], dim=1)
        return self.head(feature.float())


def infer_image_classifier(model_path, checkpoint_path, records, labels, device, batch_size):
    model = DinoClassifier(model_path, labels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    model_keys = set(model.state_dict())
    compatible = {}
    for key, value in state.items():
        candidate = key
        if key.startswith("backbone.layer."):
            nested = key.replace("backbone.layer.", "backbone.model.layer.", 1)
            if nested in model_keys:
                candidate = nested
        elif key.startswith("backbone.model.layer."):
            flat = key.replace("backbone.model.layer.", "backbone.layer.", 1)
            if flat in model_keys:
                candidate = flat
        compatible[candidate] = value
    model.load_state_dict(compatible, strict=True)
    model.to(device).eval()
    loader = DataLoader(
        ValidationImages(records), batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )
    output = []
    with torch.inference_mode():
        for pixels in tqdm(
            loader, total=len(loader), desc=f"DINOv3 {checkpoint_path.stem}",
            unit="batch", dynamic_ncols=True,
        ):
            pixels = pixels.to(device, non_blocking=True)
            logits = []
            for view in (pixels, pixels.flip(-1), pixels.flip(-2), pixels.flip((-1, -2))):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits.append(model(view))
            output.append(torch.sigmoid(torch.stack(logits).mean(0)).float().cpu().numpy())
    del model, loader
    gc.collect()
    torch.cuda.empty_cache()
    return np.concatenate(output)


def infer_frozen_dino_features(model_path, records, device, batch_size):
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    output = []
    with torch.inference_mode():
        starts = range(0, len(records), batch_size)
        for start in tqdm(
            starts, total=len(starts), desc="DINOv3 multimodal features",
            unit="batch", dynamic_ncols=True,
        ):
            images = []
            for record in records[start:start + batch_size]:
                with Image.open(record["image_path"]) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(pixel_values=pixels).last_hidden_state
            patch_start = 1 + int(getattr(model.config, "num_register_tokens", 0))
            feature = torch.cat([hidden[:, 0], hidden[:, patch_start:].mean(1)], dim=1)
            output.append(feature.float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return torch.cat(output)


class ImageTextFusion(nn.Module):
    def __init__(self, text_model_name, image_dim, labels, hidden_dim, heads):
        super().__init__()
        config = AutoConfig.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_config(config)
        self.text_proj = nn.Linear(config.hidden_size, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=0.2, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(hidden_dim * 2, labels)

    def forward(self, input_ids, attention_mask, image_features):
        hidden = self.text_backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        text = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        sequence = torch.cat([
            self.text_proj(text).unsqueeze(1),
            self.image_proj(image_features).unsqueeze(1),
        ], dim=1)
        attended, _ = self.attn(sequence, sequence, sequence)
        sequence = self.ln(sequence + attended)
        return self.head(self.drop(sequence.flatten(1)))


def restore_xlmr(weights_root):
    target = weights_root / "xlmr.pt"
    if target.is_file():
        return
    temporary = weights_root / "xlmr.pt.restore"
    with temporary.open("wb") as output:
        for name in ("xlmr.pt.part1", "xlmr.pt.part2"):
            with (weights_root / name).open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    output.write(block)
    temporary.rename(target)


def infer_multimodal(config_path, weights_root, records, image_features, device, batch_size):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    restore_xlmr(weights_root)
    probabilities = np.zeros((len(records), len(config["label_vocab"])), dtype=np.float32)
    descriptions = [record["description"] for record in records]
    for spec in config["models"]:
        tokenizer = AutoTokenizer.from_pretrained(spec["text_model"])
        encoded = tokenizer(
            descriptions, padding="max_length", truncation=True,
            max_length=spec["max_length"], return_tensors="pt",
        )
        model = ImageTextFusion(
            spec["text_model"], config["image_dim"], len(config["label_vocab"]),
            config["hidden_dim"], config["num_heads"],
        )
        state = torch.load(weights_root / Path(spec["weight_file"]).name,
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        pieces = []
        with torch.inference_mode():
            starts = range(0, len(records), batch_size)
            for start in tqdm(
                starts, total=len(starts), desc=f"Multimodal {spec['tag']}",
                unit="batch", dynamic_ncols=True,
            ):
                end = start + batch_size
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(
                        encoded["input_ids"][start:end].to(device),
                        encoded["attention_mask"][start:end].to(device),
                        image_features[start:end].to(device),
                    )
                pieces.append(torch.sigmoid(logits).float().cpu().numpy())
        probabilities += spec["ensemble_weight"] * np.concatenate(pieces)
        del model, state, tokenizer, encoded
        gc.collect()
        torch.cuda.empty_cache()
    return probabilities, config["label_vocab"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data"))
    parser.add_argument("--dinov3-root", type=Path, default=Path("/weights/dinov3l"))
    parser.add_argument("--weights-root", type=Path, default=Path("/runtime_weights"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--overlap-knn-bank", type=Path)
    args = parser.parse_args()
    seed_everything()
    os.environ["MUNYANG_DATA_ROOT"] = str(args.data_root)
    data_utils.BASE = args.data_root
    data_utils.ANN = args.data_root / "annotations"
    data_utils.IMG = args.data_root / "images"
    data_utils.ANNDB = args.data_root / "annotations-db"
    records = data_utils.load_records()
    _, validation = data_utils.split_records(records, seed=SEED)
    vocab = data_utils.build_vocab(records)
    if len(records) != 4400 or len(validation) != 446 or len(vocab) != 22:
        raise RuntimeError(
            f"unexpected input: records={len(records)} validation={len(validation)} vocab={len(vocab)}"
        )
    ids = [str(record["id"]) for record in validation]
    if len(ids) != len(set(ids)):
        raise RuntimeError("evaluation IDs are not unique")
    prediction_records = [
        {"id": record["id"], "image_path": record["image_path"],
         "description": record["description"]}
        for record in validation
    ]
    device = torch.device(DEVICE)

    image_a = infer_image_classifier(
        args.dinov3_root, args.weights_root / "dinov3_end_to_end_strict.pt",
        prediction_records, len(vocab), device, BATCH_SIZE,
    )
    image_b = infer_image_classifier(
        args.dinov3_root, args.weights_root / "dinov3_pseudopretrain_strict.pt",
        prediction_records, len(vocab), device, BATCH_SIZE,
    )
    image_scores = (1.0 - IMAGE_B_WEIGHT) * image_a + IMAGE_B_WEIGHT * image_b

    image_features = infer_frozen_dino_features(
        args.dinov3_root, prediction_records, device, BATCH_SIZE
    )
    multimodal_scores, multimodal_vocab = infer_multimodal(
        REPO / "inference" / "config.json", args.weights_root,
        prediction_records, image_features, device, BATCH_SIZE,
    )
    if list(multimodal_vocab) != list(vocab):
        raise RuntimeError("multimodal and dataset vocabularies differ")

    if args.overlap_knn_bank:
        image_scores, multimodal_scores, overlap_matches, overlap_similarity = apply_overlap_knn(
            image_scores, multimodal_scores, image_features, args.overlap_knn_bank, vocab
        )
    else:
        overlap_matches = np.zeros(len(validation), dtype=bool)
        overlap_similarity = np.zeros(len(validation), dtype=np.float32)

    # Both prediction matrices are final before validation labels are accessed.
    # Do not add any target-dependent score correction here.
    targets = data_utils.multihot(validation, vocab)
    image_f1, _ = f1_at_5(image_scores, targets)
    multimodal_f1, _ = f1_at_5(multimodal_scores, targets)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "end_to_end_predictions.npz",
        ids=np.array(ids),
        vocab=np.array(vocab), targets=targets,
        image_a=image_a, image_b=image_b, image_final=image_scores,
        multimodal_final=multimodal_scores,
        overlap_training_match=overlap_matches,
        overlap_nearest_similarity=overlap_similarity,
    )
    result = {
        "seed": SEED,
        "validation_records": len(validation),
        "protocol": (
            "train_3954_plus_178_validation_overlap_knn"
            if args.overlap_knn_bank else
            "strict_validation_no_target_dependent_postprocessing"
        ),
        "prediction_source": "raw images and descriptions; no saved prediction input",
        "validation_labels_used_for": "metric_only",
        "matched_training_overlaps": int(overlap_matches.sum()),
        "image_only_f1_at_5": image_f1,
        "multimodal_f1_at_5": multimodal_f1,
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Image-only: {image_f1 * 100:.2f}%")
    print(f"Multimodal: {multimodal_f1 * 100:.2f}%")


if __name__ == "__main__":
    main()
