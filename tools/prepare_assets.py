"""Restore split Git LFS weights and optionally verify every published asset."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def is_lfs_pointer(path):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return f.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def restore_split_weight(weights, record):
    weights = Path(weights)
    target = weights / record["file"]
    if target.is_file():
        if is_lfs_pointer(target) or target.stat().st_size != record["size"]:
            raise ValueError(f"Invalid existing weight: {target}")
        return False
    parts = [weights / part["file"] for part in record["parts"]]
    for path, spec in zip(parts, record["parts"]):
        if not path.is_file() or is_lfs_pointer(path) or path.stat().st_size != spec["size"]:
            raise FileNotFoundError(f"Run git lfs pull to download the weight: {path}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=weights, prefix="xlmr.", suffix=".restore", delete=False) as out:
            temporary = Path(out.name)
            combined = hashlib.sha256()
            for path, spec in zip(parts, record["parts"]):
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        out.write(block)
                        digest.update(block)
                        combined.update(block)
                if digest.hexdigest() != spec["sha256"]:
                    raise ValueError(f"Weight part SHA-256 mismatch: {path}")
        if temporary.stat().st_size != record["size"] or combined.hexdigest() != record["sha256"]:
            raise ValueError("Restored weight SHA-256 mismatch")
        os.replace(temporary, target)
        return True
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

def prepare(weights):
    record = json.loads((ROOT / "configs/weight_parts.json").read_text())
    return restore_split_weight(weights, record)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, default=ROOT / "assets/weights")
    p.add_argument("--verify", action="store_true", help="Verify all assets in the default assets directory")
    args = p.parse_args()
    restored = prepare(args.weights_dir)
    print("XLM-R weight:", "restored and verified" if restored else "ready")
    if args.verify:
        if args.weights_dir.resolve() != (ROOT / "assets/weights").resolve():
            p.error("--verify checks the default assets layout; omit --weights-dir")
        manifest = json.loads((ROOT / "configs/asset_manifest.json").read_text())
        for name, spec in manifest.items():
            path = ROOT / name
            if not path.is_file() or path.stat().st_size != spec["size"] or sha256(path) != spec["sha256"]:
                raise ValueError(f"Asset SHA-256 mismatch or missing file: {path}")
        print(f"Assets verified: {len(manifest)}/{len(manifest)}")
if __name__ == "__main__":
    main()
