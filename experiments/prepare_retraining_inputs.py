"""Prepare ID-only training selection from the previously fixed 178 records."""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source-bank", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with np.load(args.source_bank, allow_pickle=True) as source:
    ids = source["selected_ids"].astype(str).tolist()
if len(ids) != 178 or len(set(ids)) != 178:
    raise ValueError("expected 178 unique training-selection IDs")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"selected_ids": ids}, indent=2), encoding="utf-8")
print(f"Training selection saved: {len(ids)} IDs")
