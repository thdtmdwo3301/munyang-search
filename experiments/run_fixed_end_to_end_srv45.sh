#!/usr/bin/env bash
set -euo pipefail

root=/home/mount/SSD_A/hyobin/pattern_multilabel
output="$root/fixed_end_to_end_eval"
cache="$root/hf_cache"
mkdir -p "$output" "$cache"

docker run --rm \
  --memory=120g --memory-swap=120g \
  --gpus '"device=1"' \
  --shm-size=32g --ipc=host \
  -e HF_HOME=/hf_cache \
  -v "$root/munyang-search:/workspace:ro" \
  -v "$root/data/ETRI/orig_4.4k_260519:/data:ro" \
  -v "$root/weights/dinov3l:/weights/dinov3l:ro" \
  -v "$root/runtime_weights:/runtime_weights" \
  -v "$output:/output" \
  -v "$cache:/hf_cache" \
  learner-night:latest \
  bash -lc 'cd /workspace && python experiments/evaluate_fixed_end_to_end.py'
