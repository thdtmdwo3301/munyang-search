"""Extract DINOv2 or DINOv3 image features with every torchrun rank doing real work."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def init_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


class ImageDataset(Dataset):
    def __init__(self, records, indices, processor):
        self.records = records
        self.indices = indices
        self.processor = processor

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = self.indices[position]
        record = self.records[index]
        with Image.open(record["image_path"]) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ][0]
        return index, pixel_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="facebook/dinov2-large")
    parser.add_argument("--tag", default="dinov2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    rank, world, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    records = data_utils.load_records()
    vocab = data_utils.build_vocab(records)
    labels = data_utils.multihot(records, vocab)
    args.output.mkdir(parents=True, exist_ok=True)

    local_model = Path(args.model).exists()
    # For remote models, rank 0 downloads exactly once. Other ranks then read the shared cache.
    if rank == 0 and not local_model:
        AutoImageProcessor.from_pretrained(args.model)
        AutoModel.from_pretrained(args.model)
    if world > 1:
        dist.barrier()

    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[
        args.precision
    ]
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    indices = list(range(rank, len(records), world))
    loader = DataLoader(
        ImageDataset(records, indices, processor),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    local_indices, local_features = [], []
    with torch.inference_mode():
        for step, (batch_indices, pixels) in enumerate(loader, 1):
            pixels = pixels.to(device, non_blocking=True, dtype=dtype)
            hidden = model(pixel_values=pixels).last_hidden_state
            cls = hidden[:, 0]
            # DINOv3 inserts register tokens between CLS and patch tokens.
            patch_start = 1 + int(getattr(model.config, "num_register_tokens", 0))
            patch_mean = hidden[:, patch_start:].mean(dim=1)
            features = torch.cat([cls, patch_mean], dim=1)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            local_indices.append(batch_indices.numpy())
            local_features.append(features.cpu().numpy().astype(np.float16))
            if step % 10 == 0 or step == len(loader):
                print(
                    f"rank={rank}/{world} step={step}/{len(loader)} samples={sum(map(len, local_indices))}",
                    flush=True,
                )

    shard_path = args.output / f"{args.tag}_rank{rank:02d}.npz"
    np.savez_compressed(
        shard_path,
        indices=np.concatenate(local_indices),
        features=np.concatenate(local_features),
    )
    if world > 1:
        dist.barrier()

    if rank == 0:
        features = None
        ordered = np.empty(len(records), dtype=object)
        matrix = None
        for shard_rank in range(world):
            shard = np.load(args.output / f"{args.tag}_rank{shard_rank:02d}.npz")
            if matrix is None:
                matrix = np.empty((len(records), shard["features"].shape[1]), dtype=np.float16)
            matrix[shard["indices"]] = shard["features"]
        for i, record in enumerate(records):
            ordered[i] = record["id"]
        final_path = args.output / f"{args.tag}_all_features.npz"
        np.savez_compressed(
            final_path,
            features=matrix,
            labels=labels.astype(np.float32),
            ids=ordered,
            vocab=np.array(vocab, dtype=object),
            model=args.model,
        )
        print(f"saved={final_path} shape={matrix.shape} labels={labels.shape}", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
