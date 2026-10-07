"""Sequential fine-tuning of five released neural fusion checkpoints."""
import json
import argparse
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
config = json.loads((root / "inference/config.json").read_text())
parser = argparse.ArgumentParser()
parser.add_argument("--tags", nargs="+")
args = parser.parse_args()
for spec in config["models"]:
    tag = spec["tag"]
    if args.tags and tag not in args.tags:
        continue
    subprocess.run([
        sys.executable, str(root / "train/train_fusion_final.py"),
        "--model", spec["text_model"], "--tag", tag, "--desc_only",
        "--max_length", str(spec["max_length"]),
        "--selected-ids-json", "/inputs/selected_training_ids.json",
        "--all-features", "/inputs/dinov3l_all_features.npz",
        "--init-checkpoint", f"/runtime_weights/{tag}.pt",
        "--output-dir", "/output", "--epochs", "12", "--bs", "8",
        "--grad_accum", "2", "--lr", "5e-6", "--head_lr", "5e-5",
        "--dropout", "0.1", "--bf16", "--gpu", "0",
    ], check=True)
