#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
project_root="$(cd -- "$repo_root/.." && pwd)"

data_root="${MUNYANG_DATA_ROOT:-$project_root/data/ETRI/orig_4.4k_260519}"
dinov3_root="${MUNYANG_DINOV3_ROOT:-$project_root/weights/dinov3l}"
weights_root="${MUNYANG_RUNTIME_WEIGHTS:-$project_root/runtime_weights}"
output_root="${MUNYANG_OUTPUT_ROOT:-$project_root/fixed_end_to_end_eval}"
cache_root="${MUNYANG_HF_CACHE:-$project_root/hf_cache}"
docker_image="${MUNYANG_DOCKER_IMAGE:-learner-night:latest}"
docker_memory="${MUNYANG_DOCKER_MEMORY:-64g}"

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "Required path not found: $1" >&2
    exit 1
  fi
}

require_path "$data_root/annotations"
require_path "$data_root/images"
require_path "$dinov3_root/config.json"
require_path "$dinov3_root/model.safetensors"
for filename in \
  dinov3_end_to_end_strict.pt \
  dinov3_pseudopretrain_strict.pt \
  klue.pt kcbert.pt mbert.pt kobigbird.pt; do
  require_path "$weights_root/$filename"
done
if [[ ! -f "$weights_root/xlmr.pt" ]]; then
  require_path "$weights_root/xlmr.pt.part1"
  require_path "$weights_root/xlmr.pt.part2"
fi

if [[ -n "${MUNYANG_GPU_DEVICE:-}" ]]; then
  gpu_device="$MUNYANG_GPU_DEVICE"
else
  command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi is required to select an idle GPU." >&2
    exit 1
  }
  gpu_device="$({
    nvidia-smi \
      --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits || true
  } | awk -F',' '
    {
      gsub(/[[:space:]]/, "", $1)
      gsub(/[[:space:]]/, "", $2)
      gsub(/[[:space:]]/, "", $3)
      if (($2 + 0) <= 100 && ($3 + 0) <= 5) { print $1; exit }
    }
  ')"
  if [[ -z "$gpu_device" ]]; then
    echo "No idle GPU found (requires <=100 MiB memory and <=5% utilization)." >&2
    echo "Set MUNYANG_GPU_DEVICE explicitly only after confirming a GPU is free." >&2
    exit 1
  fi
fi

mkdir -p "$output_root" "$cache_root"

echo "GPU: $gpu_device"
echo "Data: $data_root"
echo "Weights: $weights_root"
echo "Output: $output_root"

docker run --rm \
  --memory="$docker_memory" --memory-swap="$docker_memory" \
  --gpus "device=$gpu_device" \
  --shm-size=32g --ipc=host \
  -e HF_HOME=/hf_cache \
  -v "$repo_root:/workspace:ro" \
  -v "$data_root:/data:ro" \
  -v "$dinov3_root:/weights/dinov3l:ro" \
  -v "$weights_root:/runtime_weights" \
  -v "$output_root:/output" \
  -v "$cache_root:/hf_cache" \
  "$docker_image" \
  bash -lc 'cd /workspace && python experiments/evaluate_fixed_end_to_end.py'
