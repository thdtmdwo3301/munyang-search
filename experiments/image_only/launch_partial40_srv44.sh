#!/usr/bin/env bash
set -euo pipefail

output=/home/hyobin/pattern_multilabel/outputs/dinov3_partial40_kmeans
mkdir -p "$output"

docker run --rm \
  --memory=240g --memory-swap=240g \
  --gpus '"device=0,1,2,3,4,5"' \
  --shm-size=64g --ipc=host \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e MUNYANG_DATA_ROOT=/data \
  -v /home/hyobin/pattern_multilabel/munyang-search:/workspace \
  -v /home/hyobin/pattern_multilabel/data/ETRI/orig_4.4k_260519:/data \
  -v /home/hyobin/pattern_multilabel/weights/dinov3l:/weights/dinov3l \
  -v /home/hyobin/pattern_multilabel:/host \
  munyang-image-only:20261006-cu128 \
  bash -lc 'cd /workspace && torchrun --standalone --nproc_per_node=6 \
    experiments/image_only/train_dinov3_end_to_end_ddp.py \
    --model-path /weights/dinov3l \
    --output /host/outputs/dinov3_partial40_kmeans \
    --epochs 40 --batch-size 16 \
    --backbone-lr 1e-5 --head-lr 5e-4 --unfreeze-blocks 8 \
    --selected-ids-json /host/partial_val40_results.json \
    --selection-method kmeans_representative'
