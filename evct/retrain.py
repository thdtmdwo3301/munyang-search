"""Train shared image/text/fusion modules and export a reloadable prediction config."""
import argparse
import copy
import gc
import json
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from . import build_model, build_component, load_checkpoint

def validate_records(train, val, vocab, required):
    if len(vocab) < 5 or len(set(vocab)) != len(vocab):
        raise ValueError("Invalid label vocabulary")
    ids = []
    for name, rows in (("train", train), ("validation", val)):
        if not rows:
            raise ValueError(f"{name} is empty")
        group = []
        for r in rows:
            group.append(r["id"])
            labels = r["emotions"]
            if len(labels) != 5 or len(set(labels)) != 5 or not set(labels) <= set(vocab):
                raise ValueError(f"Invalid GT labels: {r['id']}")
            if "image" in required and not Path(r["image_path"]).is_file():
                raise ValueError(f"Image missing: {r['id']}")
            if "text" in required and not r.get("description", "").strip():
                raise ValueError(f"Text missing: {r['id']}")
        if len(set(group)) != len(group):
            raise ValueError(f"Duplicate IDs in {name}")
        ids.append(set(group))
    if ids[0] & ids[1]:
        raise ValueError("Training/validation IDs overlap")

def configure_training(model, freeze_image=False, freeze_text=False):
    for name, frozen in (("image_encoder", freeze_image), ("text_encoder", freeze_text)):
        module = getattr(model, name)
        if module is not None and frozen:
            module.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters")
    return params

def training_mode(model):
    model.train()
    for module in (model.image_encoder, model.text_encoder):
        if module is not None and not any(p.requires_grad for p in module.parameters()):
            module.eval()

def save_weights(model, path):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, temp)
    temp.replace(path)

def amp(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()

class Inputs:
    def __init__(self, config, required, device, feature_encoder):
        self.config, self.required, self.device, self.feature_encoder = config, required, device, feature_encoder
        self.processor = AutoImageProcessor.from_pretrained(config["image_processor"], local_files_only=True) if "image" in required else None
        self.tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], local_files_only=True) if "text" in required else None
        self.lookup = {label: i for i, label in enumerate(config["label_vocab"])}

    def __call__(self, rows):
        inputs = {}
        if self.processor is not None:
            images = []
            for r in rows:
                with Image.open(r["image_path"]) as im:
                    images.append(im.convert("RGB"))
            pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device)
            if self.feature_encoder is None:
                inputs["pixel_values"] = pixels
            else:
                with torch.no_grad(), amp(self.device):
                    inputs["image_features"] = self.feature_encoder(pixels).float()
        if self.tokenizer is not None:
            tokens = self.tokenizer([r["description"] for r in rows], padding="max_length",
                truncation=True, max_length=self.config.get("max_length", 384), return_tensors="pt")
            inputs.update({k: tokens[k].to(self.device) for k in ("input_ids", "attention_mask")})
        targets = torch.zeros(len(rows), len(self.lookup), device=self.device)
        for i, r in enumerate(rows):
            targets[i, [self.lookup[e] for e in r["emotions"]]] = 1
        return inputs, targets

@torch.no_grad()
def evaluate(model, rows, prepare, batch_size, device):
    model.eval()
    scores, hits = [], 0
    for start in range(0, len(rows), batch_size):
        inputs, targets = prepare(rows[start:start+batch_size])
        with amp(device):
            logits = model(**inputs)
        probs = logits.float().sigmoid()
        hits += int(targets.gather(1, probs.topk(5, dim=1).indices).sum())
        scores.append(probs.cpu())
    return torch.cat(scores), hits / (len(rows) * 5)

def main():
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", type=Path)
    source.add_argument("--train-jsonl", type=Path)
    p.add_argument("--val-jsonl", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-image", action="store_true")
    p.add_argument("--freeze-text", action="store_true")
    p.add_argument("--limit-train", type=int)
    p.add_argument("--limit-val", type=int)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0:
        p.error("epochs, batch-size and lr must be positive")
    if any(v is not None and v < 1 for v in (a.limit_train, a.limit_val)):
        p.error("Sample limits must be positive")
    if bool(a.train_jsonl) != bool(a.val_jsonl):
        p.error("train-jsonl and val-jsonl must be supplied together")
    if a.output.exists() and any(a.output.iterdir()):
        p.error("Output directory is not empty")
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    from .paths import load_model_config, portable_model_config
    config = load_model_config(a.config)
    config = copy.deepcopy(config)
    for key in ("checkpoint", "image_processor", "tokenizer"):
        value = config.get(key)
        if value and (a.config.parent / value).exists():
            config[key] = str((a.config.parent / value).resolve())
    # The actual module declares its modalities; custom modules use the same contract.
    model = build_model(config["model"])
    required = set(model.classifier.required_modalities)
    if required not in ({"image"}, {"text"}, {"image", "text"}):
        raise ValueError("Unsupported required_modalities")
    if config.get("checkpoint"):
        load_checkpoint(model, config["checkpoint"], config.get("checkpoint_layout", "modular"))
    elif config["model"].get("text_encoder") and not config["model"]["text_encoder"].get("pretrained", False):
        raise ValueError("Text backbone has no pretrained initialization")
    if a.data_root:
        from train import dataset
        dataset.BASE = a.data_root
        dataset.ANN, dataset.IMG, dataset.ANNDB = (a.data_root / name for name in ("annotations", "images", "annotations-db"))
        train, val = dataset.split_records(dataset.load_records(), seed=a.seed)
    else:
        def read_jsonl(path):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for r in rows:
                if "image_path" in r and not Path(r["image_path"]).is_absolute():
                    r["image_path"] = str((path.parent / r["image_path"]).resolve())
            return rows
        train, val = read_jsonl(a.train_jsonl), read_jsonl(a.val_jsonl)
    validate_records(train, val, config["label_vocab"], required)
    train, val = train[:a.limit_train], val[:a.limit_val]
    feature = build_component(config.get("feature_image_encoder"))
    if feature is not None:
        if model.image_encoder is not None or "image" not in required:
            raise ValueError("External image encoder conflicts with model")
        feature.requires_grad_(False).to(a.device).eval()
    params = configure_training(model, a.freeze_image, a.freeze_text)
    model.to(a.device)
    prepare = Inputs(config, required, a.device, feature)
    optimizer = torch.optim.AdamW(params, lr=a.lr)
    initial = {name: value.detach().cpu().clone() for name, value in model.named_parameters() if value.requires_grad}
    a.output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "arguments": {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
        "train_ids": [r["id"] for r in train], "validation_ids": [r["id"] for r in val],
        "loss": "BCEWithLogitsLoss", "memory": "off", "text_field": "description",
        "limited_run": a.limit_train is not None or a.limit_val is not None}
    (a.output / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"학습 {len(train):,}건 · 검증 {len(val):,}건", flush=True)
    history = []
    for epoch in range(a.epochs):
        training_mode(model)
        indices = list(range(len(train)))
        random.Random(a.seed + epoch).shuffle(indices)
        total = 0.
        for start in range(0, len(indices), a.batch_size):
            rows = [train[i] for i in indices[start:start+a.batch_size]]
            inputs, targets = prepare(rows)
            optimizer.zero_grad(set_to_none=True)
            with amp(a.device):
                logits = model(**inputs)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        scores, f1 = evaluate(model, val, prepare, a.batch_size, a.device)
        history.append({"epoch": epoch+1, "loss": total/len(train), "f1_at_5": f1})
        print(f"Epoch {epoch+1}/{a.epochs} · Loss {total/len(train):.4f} · F1@5 {100*f1:.2f}%", flush=True)
    changed = [name for name, value in model.named_parameters() if name in initial and not torch.equal(initial[name], value.detach().cpu())]
    if not changed:
        raise RuntimeError("No model weights changed")
    save_weights(model, a.output / "model.pt")
    export = copy.deepcopy(config)
    export.update(checkpoint="model.pt", checkpoint_layout="modular")
    (a.output / "predict.json").write_text(json.dumps(portable_model_config(export, a.output.resolve()), ensure_ascii=False, indent=2))
    del optimizer, params, initial, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = build_model(export["model"])
    load_checkpoint(reloaded, a.output / "model.pt", "modular")
    reloaded.to(a.device)
    again, reloaded_f1 = evaluate(reloaded, val, prepare, a.batch_size, a.device)
    exact = torch.equal(scores, again)
    report = {"status": "PASS" if exact else "FAIL", "train_count": len(train), "validation_count": len(val),
        "changed_parameter_count": len(changed), "history": history, "reload_f1_at_5": reloaded_f1,
        "reload_scores_exact": exact, "max_score_difference": float((scores-again).abs().max()),
        "limited_run": manifest["limited_run"], "memory": "off"}
    (a.output / "verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (a.output / "predictions.jsonl").open("w") as f:
        for r, row in zip(val, again):
            labels = [export["label_vocab"][i] for i in row.topk(5).indices.tolist()]
            f.write(json.dumps({"id": r["id"], "predicted_top5": labels,
                "correct_count": len(set(labels) & set(r["emotions"]))}, ensure_ascii=False)+"\n")
    if not exact:
        raise RuntimeError("Reloaded predictions differ")
    print("저장 후 재추론: 일치", flush=True)
    print(f"결과: {a.output}", flush=True)

if __name__ == "__main__":
    main()
