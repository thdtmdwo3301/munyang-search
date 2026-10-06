"""Strict image+caption Qwen3-VL classifier on the original 3,954/446 split."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size(), local_rank


class MultimodalDataset(Dataset):
    def __init__(self, records, vocab):
        self.records = records
        self.l2i = {label: index for index, label in enumerate(vocab)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as source:
            image = source.convert("RGB")
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            image = image.copy()
        target = torch.zeros(len(self.l2i), dtype=torch.float32)
        for label in record["emotions"]:
            target[self.l2i[label]] = 1.0
        return image, record["description"], target


class Collator:
    def __init__(self, processor, vocab):
        self.processor = processor
        self.processor.tokenizer.padding_side = "left"
        labels = ", ".join(vocab)
        self.instruction = (
            "이미지와 원설명문을 함께 보고 전통 문양의 감성 라벨을 분류하세요. "
            f"후보 라벨은 {labels} 입니다. 원설명문: "
        )

    def __call__(self, batch):
        images, descriptions, targets = zip(*batch)
        texts = []
        for image, description in zip(images, descriptions):
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.instruction + description},
            ]}]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        encoded = self.processor(
            text=texts, images=list(images), padding=True, return_tensors="pt")
        return encoded, torch.stack(targets)


class QwenMultimodalClassifier(nn.Module):
    def __init__(self, model_path, labels, unfreeze_text_blocks, dropout):
        super().__init__()
        full = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        self.base = full.model
        dim = full.config.text_config.hidden_size
        full.model = None
        del full
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        if unfreeze_text_blocks:
            for block in self.base.language_model.layers[-unfreeze_text_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, labels),
        )

    def forward(self, inputs):
        output = self.base(**inputs, use_cache=False, return_dict=True)
        return self.head(output.last_hidden_state[:, -1].float())


def f1_at_5(scores, targets):
    top5 = np.argpartition(-scores, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, top5, axis=1).sum() / (len(targets) * 5.0))


def move_inputs(inputs, device):
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.inference_mode():
        for inputs, target in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(move_inputs(inputs, device))
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(target.numpy())
    scores, targets = np.concatenate(scores), np.concatenate(targets)
    return f1_at_5(scores, targets), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--unfreeze-text-blocks", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    records = data_utils.load_records()
    train_records, val_records = data_utils.split_records(records)
    vocab = data_utils.build_vocab(records)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    collator = Collator(processor, vocab)
    train_data = MultimodalDataset(train_records, vocab)
    val_data = MultimodalDataset(val_records, vocab)
    sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, num_workers=4,
        pin_memory=True, collate_fn=collator, persistent_workers=True)
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
            pin_memory=True, collate_fn=collator, persistent_workers=True)

    classifier = QwenMultimodalClassifier(
        args.model_path, len(vocab), args.unfreeze_text_blocks, 0.2).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        classifier.load_state_dict(checkpoint["state_dict"], strict=False)
    model = DDP(classifier, device_ids=[local_rank], find_unused_parameters=True)
    backbone = [parameter for name, parameter in model.named_parameters()
                if "base" in name and parameter.requires_grad]
    head = [parameter for name, parameter in model.named_parameters()
            if "head" in name and parameter.requires_grad]
    groups = [{"params": head, "lr": args.head_lr, "weight_decay": 0.01}]
    if backbone:
        groups.insert(0, {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(groups)
    steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, len(train_loader)), steps)
    criterion = nn.BCEWithLogitsLoss()
    best = (-1.0, -1, None, None)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(move_inputs(inputs, device))
                loss = criterion(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        dist.barrier()
        if rank == 0:
            value, probabilities = evaluate(model.module, val_loader, device)
            print(f"epoch={epoch}/{args.epochs} loss={total/len(train_loader):.6f} "
                  f"strict_val_F1@5={value:.4f}", flush=True)
            if value > best[0]:
                best = (value, epoch,
                        {key: value.detach().cpu() for key, value in model.module.state_dict().items()
                         if key.startswith("head") or value.requires_grad}, probabilities)
        dist.barrier()
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best[2], "val_f1_at_5": best[0],
                    "best_epoch": best[1], "vocab": vocab, "args": vars(args)},
                   args.output / "qwen3vl_multimodal_strict.pt")
        np.savez_compressed(
            args.output / "qwen3vl_multimodal_strict_val.npz", probabilities=best[3],
            targets=data_utils.multihot(val_records, vocab),
            ids=np.array([record["id"] for record in val_records]), vocab=np.array(vocab))
        result = {"metric": "F1@5", "input": "image_and_caption",
                  "validation_in_training": False, "train_records": len(train_records),
                  "validation_records": len(val_records), "best_f1_at_5": best[0],
                  "best_epoch": best[1]}
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
