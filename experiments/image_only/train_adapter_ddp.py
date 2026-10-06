"""Train an image-only MLP with DDP and evaluate strict and permitted protocols."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def init_distributed():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


class Adapter(nn.Module):
    def __init__(self, dim, labels, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, labels),
        )

    def forward(self, x):
        return self.net(x)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        probs_pos = torch.sigmoid(logits)
        probs_neg = 1.0 - probs_pos
        if self.clip:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp(min=self.eps))
        pt = probs_pos * targets + probs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        return -(((1.0 - pt) ** gamma) * (loss_pos + loss_neg)).sum(dim=1).mean()


def f1_at_5(scores, labels):
    top = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    return float(np.take_along_axis(labels, top, axis=1).sum() / (len(labels) * 5.0))


def cache_scores(query, bank, bank_labels, k=1, temperature=0.02, exclude=None):
    similarity = query @ bank.T
    if exclude is not None:
        similarity[np.arange(len(query)), exclude] = -np.inf
    k = min(k, bank.shape[0])
    nearest = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    nearest_sim = np.take_along_axis(similarity, nearest, axis=1)
    weights = np.exp((nearest_sim - nearest_sim.max(axis=1, keepdims=True)) / temperature)
    weights /= weights.sum(axis=1, keepdims=True)
    return (bank_labels[nearest] * weights[..., None]).sum(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["strict", "train_plus_val"], required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["asl", "bce"], default="asl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world, local_rank = init_distributed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda", local_rank)

    payload = np.load(args.features, allow_pickle=True)
    features = payload["features"].astype(np.float32)
    labels = payload["labels"].astype(np.float32)
    ids = payload["ids"].astype(str)
    records = data_utils.load_records()
    train_records, val_records = data_utils.split_records(records)
    id_to_index = {sample_id: i for i, sample_id in enumerate(ids)}
    train_idx = np.array([id_to_index[r["id"]] for r in train_records])
    val_idx = np.array([id_to_index[r["id"]] for r in val_records])
    fit_idx = train_idx if args.protocol == "strict" else np.arange(len(ids))

    x = torch.from_numpy(features[fit_idx])
    y = torch.from_numpy(labels[fit_idx])
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True)

    model = Adapter(features.shape[1], labels.shape[1], args.hidden, args.dropout).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = AsymmetricLoss() if args.loss == "asl" else nn.BCEWithLogitsLoss()
    x_val = torch.from_numpy(features[val_idx]).to(device)
    best = (-1.0, None, -1)

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.inference_mode():
                val_scores = torch.sigmoid(model.module(x_val)).cpu().numpy()
            score = f1_at_5(val_scores, labels[val_idx])
            if rank == 0:
                print(
                    f"protocol={args.protocol} epoch={epoch}/{args.epochs} "
                    f"loss={running / max(1, len(loader)):.4f} val_F1@5={score:.4f}",
                    flush=True,
                )
                if score > best[0]:
                    best = (score, {k: v.detach().cpu() for k, v in model.module.state_dict().items()}, epoch)
        dist.barrier()

    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        checkpoint = args.output / f"adapter_{args.protocol}.pt"
        torch.save(
            {
                "state_dict": best[1],
                "protocol": args.protocol,
                "val_f1_at_5": best[0],
                "epoch": best[2],
                "feature_model": str(payload["model"]),
            },
            checkpoint,
        )

        x_all = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
        strict_cache = cache_scores(x_all[val_idx], x_all[train_idx], labels[train_idx], k=1)
        permitted_cache = cache_scores(x_all[val_idx], x_all, labels, k=1)
        # Leave the exact query out while still allowing the other validation labels in the bank.
        loo_cache = cache_scores(x_all[val_idx], x_all, labels, k=1, exclude=val_idx)
        results = {
            "metric": "F1@5",
            "validation_samples": int(len(val_idx)),
            "train_samples": int(len(train_idx)),
            "protocol": args.protocol,
            "mlp_best_f1_at_5": best[0],
            "mlp_best_epoch": best[2],
            "strict_train_only_cache_k1_f1_at_5": f1_at_5(strict_cache, labels[val_idx]),
            "train_plus_val_cache_k1_f1_at_5": f1_at_5(permitted_cache, labels[val_idx]),
            "train_plus_val_leave_self_out_cache_k1_f1_at_5": f1_at_5(loo_cache, labels[val_idx]),
            "note": "train_plus_val includes the 446 validation labels by explicit user permission; it is not an independent estimate.",
        }
        result_path = args.output / f"results_{args.protocol}.json"
        result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
