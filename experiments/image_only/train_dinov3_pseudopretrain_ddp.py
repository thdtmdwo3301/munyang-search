"""DINOv3 visual pseudo-pretraining followed by strict 3,954-record fine-tuning.

The external 449k image corpus is labelled by Qwen3-VL adjectives.  Those
adjectives are mapped to the fixed 22-label task vocabulary, then used only for
visual representation warm-up.  The original 446 validation images and labels
are never included in either training stage.
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
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from transformers import AutoModel, get_cosine_schedule_with_warmup


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Conservative semantic aliases. Substring matching is intentional because the
# source contains inflected/noisy adjective forms (for example, "우아로운").
ALIASES = {
    "강한": ("강한", "강렬", "힘찬", "힘있는", "위압", "단단", "견고"),
    "고급스러운": ("고급", "고귀", "품격", "세련"),
    "고전적인": ("고전", "전통", "오래된", "고대"),
    "귀여운": ("귀여", "깜찍"),
    "깔끔한": ("깔끔", "단정", "정돈", "정제"),
    "단순한": ("단순", "심플", "간결"),
    "독특한": ("독특", "유니크", "독창", "특이"),
    "부드러운": ("부드러", "온화", "포근"),
    "섬세한": ("섬세", "정교", "세밀", "세심"),
    "소박한": ("소박", "수수"),
    "신비한": ("신비", "오묘", "영적", "신성"),
    "우아한": ("우아", "단아"),
    "원초적인": ("원초", "원시", "본능", "야생"),
    "유려한": ("유려", "흐르는", "유동"),
    "익살스러운": ("익살", "유쾌", "재미", "해학"),
    "자연적인": ("자연", "유기적"),
    "장식적인": ("장식", "꾸민", "문양적"),
    "조화로운": ("조화", "균형"),
    "풍성한": ("풍성", "풍부", "다채", "복잡"),
    "현대적인": ("현대", "모던", "미래적"),
    "화려한": ("화려", "찬란", "눈부신"),
    "활동적인": ("활동", "역동", "경쾌", "생동", "활기"),
}


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size(), local_rank


def mapped_labels(adjectives, vocab):
    found = []
    for label in vocab:
        aliases = ALIASES.get(label, (label,))
        if any(alias in adjective for alias in aliases for adjective in adjectives):
            found.append(label)
    return found


class OfficialDataset(Dataset):
    def __init__(self, records, vocab, transform):
        self.records = records
        self.transform = transform
        self.l2i = {label: i for i, label in enumerate(vocab)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        target = torch.zeros(len(self.l2i), dtype=torch.float32)
        for label in record["emotions"]:
            target[self.l2i[label]] = 1.0
        return pixels, target


class PseudoDataset(Dataset):
    def __init__(self, image_root, jsonl, vocab, transform, excluded_ids):
        self.transform = transform
        self.l2i = {label: i for i, label in enumerate(vocab)}
        self.rows = []
        with open(jsonl, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row["identifier"] in excluded_ids:
                    continue
                path = image_root / row["identifier"]
                labels = mapped_labels(row.get("adjectives", []), vocab)
                if labels and path.is_file():
                    self.rows.append((path, labels))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, labels = self.rows[index]
        with Image.open(path) as image:
            pixels = self.transform(image.convert("RGB"))
        target = torch.zeros(len(self.l2i), dtype=torch.float32)
        for label in labels:
            target[self.l2i[label]] = 1.0
        return pixels, target


class DinoClassifier(nn.Module):
    def __init__(self, model_path, labels, unfreeze_blocks, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in self.backbone.layer[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad_(True)
        dim = self.backbone.config.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, labels),
        )

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        patch_start = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        feature = torch.cat([hidden[:, 0], hidden[:, patch_start:].mean(1)], dim=1)
        return self.head(feature.float())


def make_optimizer(model, backbone_lr, head_lr):
    backbone = [
        p for name, p in model.module.named_parameters()
        if name.startswith("backbone") and p.requires_grad
    ]
    head = [
        p for name, p in model.module.named_parameters()
        if name.startswith("head") and p.requires_grad
    ]
    return torch.optim.AdamW([
        {"params": backbone, "lr": backbone_lr, "weight_decay": 0.05},
        {"params": head, "lr": head_lr, "weight_decay": 0.01},
    ])


def train_stage(model, loader, sampler, optimizer, scheduler, epochs, device, rank, negative_weight):
    for epoch in range(1, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total = 0.0
        for pixels, targets in loader:
            pixels = pixels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixels)
                loss_matrix = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
                weights = targets + (1.0 - targets) * negative_weight
                loss = (loss_matrix * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        dist.barrier()
        if rank == 0:
            print(f"pseudo_epoch={epoch}/{epochs} loss={total / len(loader):.6f}", flush=True)
        dist.barrier()


def f1_at_5(scores, targets):
    top = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, top, axis=1).sum() / (len(targets) * 5.0))


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.inference_mode():
        for pixels, target in loader:
            pixels = pixels.to(device, non_blocking=True)
            logits = []
            for view in (pixels, pixels.flip(-1), pixels.flip(-2), pixels.flip((-1, -2))):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits.append(model(view))
            scores.append(torch.sigmoid(torch.stack(logits).mean(0)).float().cpu().numpy())
            targets.append(target.numpy())
    scores, targets = np.concatenate(scores), np.concatenate(targets)
    return f1_at_5(scores, targets), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/weights/dinov3l")
    parser.add_argument("--pseudo-image-root", type=Path, required=True)
    parser.add_argument("--pseudo-jsonl", type=Path, required=True)
    parser.add_argument("--exclusion-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pseudo-epochs", type=int, default=2)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--pseudo-batch-size", type=int, default=24)
    parser.add_argument("--finetune-batch-size", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=8)
    parser.add_argument("--pseudo-backbone-lr", type=float, default=2e-6)
    parser.add_argument("--pseudo-head-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--finetune-head-lr", type=float, default=3e-4)
    parser.add_argument("--pseudo-negative-weight", type=float, default=0.1)
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
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.55, 1.0), ratio=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.25, 0.15, 0.05)], p=0.5),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # The manifest is produced before training by audit_external_duplicates.py.
    excluded_ids = set()
    if args.exclusion_manifest:
        exclusion = json.loads(args.exclusion_manifest.read_text(encoding="utf-8"))
        excluded_ids = set(exclusion["external_validation_duplicates"])
    pseudo_dataset = PseudoDataset(
        args.pseudo_image_root, args.pseudo_jsonl, vocab, train_transform, excluded_ids
    )
    train_dataset = OfficialDataset(train_records, vocab, train_transform)
    val_dataset = OfficialDataset(val_records, vocab, val_transform)
    if rank == 0:
        print(json.dumps({
            "pseudo_records_after_validation_hash_exclusion": len(pseudo_dataset),
            "official_train_records": len(train_dataset),
            "validation_records": len(val_dataset),
            "excluded_external_validation_duplicates": len(excluded_ids),
            "vocab": vocab,
        }, ensure_ascii=False), flush=True)

    pseudo_sampler = DistributedSampler(pseudo_dataset, world, rank, shuffle=True, seed=args.seed)
    train_sampler = DistributedSampler(train_dataset, world, rank, shuffle=True, seed=args.seed)
    pseudo_loader = DataLoader(
        pseudo_dataset, batch_size=args.pseudo_batch_size, sampler=pseudo_sampler,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.finetune_batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = None
    if rank == 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.finetune_batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    model = DDP(
        DinoClassifier(args.model_path, len(vocab), args.unfreeze_blocks, 0.2).to(device),
        device_ids=[local_rank],
    )
    optimizer = make_optimizer(model, args.pseudo_backbone_lr, args.pseudo_head_lr)
    steps = len(pseudo_loader) * args.pseudo_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, steps // 20), steps)
    train_stage(
        model, pseudo_loader, pseudo_sampler, optimizer, scheduler,
        args.pseudo_epochs, device, rank, args.pseudo_negative_weight,
    )

    optimizer = make_optimizer(model, args.finetune_backbone_lr, args.finetune_head_lr)
    steps = len(train_loader) * args.finetune_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, len(train_loader)), steps)
    criterion = nn.BCEWithLogitsLoss()
    best = (-1.0, -1, None, None)
    for epoch in range(1, args.finetune_epochs + 1):
        train_sampler.set_epoch(1000 + epoch)
        model.train()
        total = 0.0
        for pixels, targets in train_loader:
            pixels = pixels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(pixels), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        dist.barrier()
        if rank == 0:
            score, probabilities = evaluate(model.module, val_loader, device)
            print(
                f"finetune_epoch={epoch}/{args.finetune_epochs} "
                f"loss={total / len(train_loader):.6f} strict_val_F1@5={score:.4f}",
                flush=True,
            )
            if score > best[0]:
                best = (
                    score, epoch,
                    {key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                    probabilities,
                )
        dist.barrier()

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best[2], "val_f1_at_5": best[0], "best_epoch": best[1],
            "vocab": vocab, "args": vars(args),
        }, args.output / "dinov3_pseudopretrain_strict.pt")
        np.savez_compressed(
            args.output / "dinov3_pseudopretrain_strict_val.npz",
            probabilities=best[3], targets=data_utils.multihot(val_records, vocab),
            ids=np.array([record["id"] for record in val_records]), vocab=np.array(vocab),
        )
        result = {
            "metric": "F1@5", "validation_in_training": False,
            "external_pseudo_records": len(pseudo_dataset),
            "official_train_records": len(train_dataset),
            "validation_records": len(val_dataset),
            "best_f1_at_5": best[0], "best_epoch": best[1],
        }
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
