"""Strict image-only classifier on the Qwen3-VL vision tower.

The language model is discarded after loading.  Only image pixels are accepted
by the dataset and forward pass.  The original 446 validation records are used
for evaluation/model selection only.
"""

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


class ImageDataset(Dataset):
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
        return image, target


class Collator:
    def __init__(self, processor):
        self.image_processor = processor.image_processor

    def __call__(self, batch):
        images, targets = zip(*batch)
        encoded = self.image_processor(images=list(images), return_tensors="pt")
        return encoded["pixel_values"], encoded["image_grid_thw"], torch.stack(targets)


class VisionClassifier(nn.Module):
    def __init__(self, model_path, labels, unfreeze_blocks, dropout):
        super().__init__()
        full = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa", low_cpu_mem_usage=True,
        )
        self.visual = full.model.visual
        full.model.visual = None
        del full
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for block in self.visual.blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for module in (self.visual.merger, self.visual.deepstack_merger_list):
            for parameter in module.parameters():
                parameter.requires_grad = True
        dim = self.visual.config.out_hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, labels),
        )

    def forward(self, pixel_values, image_grid_thw):
        output = self.visual(
            pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw,
            return_dict=True,
        )
        sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        chunks = torch.split(output.pooler_output, sizes)
        pooled = torch.stack([chunk.mean(0) for chunk in chunks])
        return self.head(pooled.float())


def f1_at_5(scores, targets):
    top5 = np.argpartition(-scores, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, top5, axis=1).sum() / (len(targets) * 5.0))


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.inference_mode():
        for pixels, grid, target in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels.to(device), grid.to(device))
            scores.append(torch.sigmoid(logits).float().cpu().numpy())
            targets.append(target.numpy())
    scores, targets = np.concatenate(scores), np.concatenate(targets)
    return f1_at_5(scores, targets), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
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
    collate = Collator(processor)
    train_data, val_data = ImageDataset(train_records, vocab), ImageDataset(val_records, vocab)
    sampler = DistributedSampler(train_data, world, rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, sampler=sampler, num_workers=4,
        pin_memory=True, collate_fn=collate, persistent_workers=True,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_data, batch_size=args.batch_size, shuffle=False, num_workers=4,
            pin_memory=True, collate_fn=collate, persistent_workers=True,
        )

    model = DDP(
        VisionClassifier(args.model_path, len(vocab), args.unfreeze_blocks, 0.2).to(device),
        device_ids=[local_rank], find_unused_parameters=True,
    )
    backbone = [parameter for name, parameter in model.named_parameters()
                if "visual" in name and parameter.requires_grad]
    head = [parameter for name, parameter in model.named_parameters()
            if "head" in name and parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.05},
        {"params": head, "lr": args.head_lr, "weight_decay": 0.01},
    ])
    steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, len(train_loader)), steps)
    criterion = nn.BCEWithLogitsLoss()
    best = (-1.0, -1, None, None)
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for pixels, grid, targets in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels.to(device), grid.to(device))
                loss = criterion(logits, targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        dist.barrier()
        if rank == 0:
            score, probabilities = evaluate(model.module, val_loader, device)
            print(f"epoch={epoch}/{args.epochs} loss={total/len(train_loader):.6f} "
                  f"strict_val_F1@5={score:.4f}", flush=True)
            if score > best[0]:
                best = (score, epoch,
                        {key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                        probabilities)
        dist.barrier()

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best[2], "val_f1_at_5": best[0],
                    "best_epoch": best[1], "vocab": vocab, "args": vars(args)},
                   args.output / "qwen3vl_vision_strict.pt")
        np.savez_compressed(
            args.output / "qwen3vl_vision_strict_val.npz", probabilities=best[3],
            targets=data_utils.multihot(val_records, vocab),
            ids=np.array([record["id"] for record in val_records]), vocab=np.array(vocab),
        )
        result = {"metric": "F1@5", "input": "image_only",
                  "validation_in_training": False, "train_records": len(train_records),
                  "validation_records": len(val_records), "best_f1_at_5": best[0],
                  "best_epoch": best[1]}
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
