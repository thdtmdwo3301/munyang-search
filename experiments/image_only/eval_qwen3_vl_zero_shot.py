"""Strict image-only Qwen3-VL evaluation, sharded over eight GPUs."""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def init_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    return dist.get_rank(), dist.get_world_size(), local_rank


def parse_labels(text, vocab, fallback):
    found = []
    positions = []
    for label in vocab:
        pos = text.find(label)
        if pos >= 0:
            positions.append((pos, label))
    for _, label in sorted(positions):
        if label not in found:
            found.append(label)
    for label in fallback:
        if len(found) == 5:
            break
        if label not in found:
            found.append(label)
    return found[:5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    rank, world, local_rank = init_distributed()

    if rank == 0:
        snapshot_download(args.model)
    dist.barrier()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(local_rank).eval()

    records = data_utils.load_records()
    train_records, val_records = data_utils.split_records(records)
    vocab = data_utils.build_vocab(records)
    counts = Counter(label for record in train_records for label in record["emotions"])
    fallback = [label for label, _ in counts.most_common()]
    labels_text = ", ".join(vocab)
    prompt = (
        "이미지만 보고 전통 문양의 감성 라벨을 예측하세요. "
        "다음 후보 22개 중 정확히 5개만 선택하세요: "
        f"{labels_text}. "
        "쉼표로 구분한 라벨 5개만 출력하고 설명은 쓰지 마세요."
    )
    shard = list(range(rank, len(val_records), world))
    rows = []
    for step, index in enumerate(shard, 1):
        record = val_records[index]
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(local_rank)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        predicted = parse_labels(raw, vocab, fallback)
        rows.append({
            "index": index,
            "id": record["id"],
            "predicted": predicted,
            "ground_truth": record["emotions"],
            "raw": raw,
        })
        if step % 10 == 0 or step == len(shard):
            print(f"rank={rank}/{world} {step}/{len(shard)}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    shard_path = args.output / f"qwen3_vl_zero_shot_rank{rank:02d}.jsonl"
    with shard_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dist.barrier()

    if rank == 0:
        merged = []
        for shard_rank in range(world):
            path = args.output / f"qwen3_vl_zero_shot_rank{shard_rank:02d}.jsonl"
            merged.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        merged.sort(key=lambda row: row["index"])
        overlap = sum(len(set(row["predicted"]) & set(row["ground_truth"])) for row in merged)
        score = overlap / (len(merged) * 5.0)
        result = {
            "metric": "F1@5",
            "protocol": "strict train 3954 / validation 446",
            "model": args.model,
            "validation_samples": len(merged),
            "f1_at_5": score,
        }
        (args.output / "results_qwen3_vl_zero_shot.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

