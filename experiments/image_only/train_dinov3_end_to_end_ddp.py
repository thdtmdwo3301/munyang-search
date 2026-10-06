"""Strict DINOv3-L end-to-end multilabel fine-tuning with eight-GPU DDP."""

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
from torchvision import transforms
from transformers import AutoModel, get_cosine_schedule_with_warmup


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size(), local_rank


class PatternDataset(Dataset):
    def __init__(self, records, vocab, transform):
        self.records = records
        self.transform = transform
        self.l2i = {label: i for i, label in enumerate(vocab)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            image = self.transform(image.convert("RGB"))
        target = torch.zeros(len(self.l2i), dtype=torch.float32)
        for label in record["emotions"]:
            target[self.l2i[label]] = 1.0
        return image, target


class DinoClassifier(nn.Module):
    def __init__(self, model_path, labels, unfreeze_blocks, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        blocks = self.backbone.layer
        for block in blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        for parameter in self.backbone.norm.parameters():
            parameter.requires_grad_(True)
        dim = self.backbone.config.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, labels),
        )

    def forward(self, pixels):
        hidden = self.backbone(pixel_values=pixels).last_hidden_state
        patch_start = 1 + int(getattr(self.backbone.config, "num_register_tokens", 0))
        feature = torch.cat([hidden[:, 0], hidden[:, patch_start:].mean(dim=1)], dim=1)
        return self.head(feature.float())


def f1_at_5(scores, targets):
    top = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, top, axis=1).sum() / (len(targets) * 5.0))


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.inference_mode():
        for pixels, target in loader:
            pixels = pixels.to(device, non_blocking=True)
            views = [pixels, pixels.flip(-1), pixels.flip(-2), pixels.flip((-1, -2))]
            logits = []
            for view in views:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits.append(model(view))
            scores.append(torch.sigmoid(torch.stack(logits).mean(0)).float().cpu().numpy())
            targets.append(target.numpy())
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    return f1_at_5(scores, targets), scores, targets


def load_selected_ids(path, method):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {str(value) for value in payload}
    if "selected_ids" in payload:
        return {str(value) for value in payload["selected_ids"]}
    return {str(value) for value in payload["methods"][method]["selected_ids"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/weights/dinov3l")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-ids-json")
    parser.add_argument("--selection-method", default="kmeans_representative")
    args = parser.parse_args()
    rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    records = data_utils.load_records()
    train_records, val_records = data_utils.split_records(records)
    selected_ids = set()
    if args.selected_ids_json:
        selected_ids = load_selected_ids(args.selected_ids_json, args.selection_method)
        selected_records = [record for record in val_records if str(record["id"]) in selected_ids]
        holdout_records = [record for record in val_records if str(record["id"]) not in selected_ids]
        if len(selected_records) != len(selected_ids):
            raise ValueError(
                f"selected id mismatch: found={len(selected_records)} requested={len(selected_ids)}"
            )
        train_records = train_records + selected_records
    else:
        selected_records = []
        holdout_records = val_records
    vocab = data_utils.build_vocab(records)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(30)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.2)], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    train_dataset = PatternDataset(train_records, vocab, train_transform)
    val_dataset = PatternDataset(val_records, vocab, val_transform)
    holdout_dataset = PatternDataset(holdout_records, vocab, val_transform)
    sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4,
        pin_memory=True, persistent_workers=True, drop_last=False,
    )
    val_loader = None
    holdout_loader = None
    if rank == 0:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        holdout_loader = DataLoader(holdout_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DinoClassifier(
        args.model_path, len(vocab), args.unfreeze_blocks, args.dropout
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    backbone = [p for n, p in model.module.named_parameters() if n.startswith("backbone") and p.requires_grad]
    head = [p for n, p in model.module.named_parameters() if n.startswith("head") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.05},
        {"params": head, "lr": args.head_lr, "weight_decay": 0.01},
    ])
    updates_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, updates_per_epoch * 2),
        num_training_steps=updates_per_epoch * args.epochs,
    )
    criterion = nn.BCEWithLogitsLoss()
    best = (-1.0, -1.0, -1, None, None)

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for step, (pixels, targets) in enumerate(train_loader, 1):
            pixels = pixels.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(pixels), targets) / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += loss.item() * args.grad_accum
        dist.barrier()
        if rank == 0:
            full_score, probabilities, targets = evaluate(model.module, val_loader, device)
            holdout_score, _, _ = evaluate(model.module, holdout_loader, device)
            print(
                f"epoch={epoch}/{args.epochs} loss={loss_sum / len(train_loader):.5f} "
                f"full_val_F1@5={full_score:.4f} holdout_F1@5={holdout_score:.4f}", flush=True,
            )
            if holdout_score > best[0]:
                best = (
                    holdout_score,
                    full_score,
                    epoch,
                    {key: value.detach().cpu() for key, value in model.module.state_dict().items()},
                    probabilities,
                )
        dist.barrier()

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best[3],
            "holdout_f1_at_5": best[0],
            "full_validation_f1_at_5": best[1],
            "best_epoch": best[2],
            "vocab": vocab,
            "args": vars(args),
        }, args.output / "dinov3_end_to_end_partial_val.pt")
        np.savez_compressed(
            args.output / "dinov3_end_to_end_partial_val.npz",
            probabilities=best[4],
            targets=data_utils.multihot(val_records, vocab),
            ids=np.array([record["id"] for record in val_records]),
            vocab=np.array(vocab),
        )
        result = {
            "metric": "F1@5",
            "protocol": f"train {len(train_records)} including {len(selected_records)} validation / full validation 446 / untouched holdout {len(holdout_records)}",
            "validation_in_training": bool(selected_records),
            "selected_validation_records": len(selected_records),
            "untouched_holdout_records": len(holdout_records),
            "full_validation_f1_at_5": best[1],
            "holdout_f1_at_5": best[0],
            "best_epoch": best[2],
            "unfreeze_blocks": args.unfreeze_blocks,
        }
        (args.output / "results_dinov3_end_to_end_partial_val.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
