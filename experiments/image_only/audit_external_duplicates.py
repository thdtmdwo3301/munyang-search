"""Find exact external copies of original train/validation images by SHA-256."""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return str(path), value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    args = parser.parse_args()

    records = data_utils.load_records()
    train_records, val_records = data_utils.split_records(records)
    train_hashes = {digest(record["image_path"])[1] for record in train_records}
    val_hashes = {digest(record["image_path"])[1] for record in val_records}
    paths = sorted(
        path for path in args.external_image_root.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    external_train_duplicates = []
    external_validation_duplicates = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for path, image_hash in pool.map(digest, paths, chunksize=128):
            identifier = Path(path).name
            if image_hash in val_hashes:
                external_validation_duplicates.append(identifier)
            elif image_hash in train_hashes:
                external_train_duplicates.append(identifier)

    result = {
        "external_images_scanned": len(paths),
        "official_train_images": len(train_records),
        "official_validation_images": len(val_records),
        "external_train_duplicates": external_train_duplicates,
        "external_validation_duplicates": external_validation_duplicates,
        "validation_duplicates_excluded_from_training": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items()}))


if __name__ == "__main__":
    main()
