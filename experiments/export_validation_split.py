"""Materialize a validation split from an ID manifest.

Raw ETRI images and annotations are intentionally not committed to Git.  This
utility creates a portable evaluation directory from the separately versioned
ID manifest on a machine that already has the authorized dataset.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transfer(source, destination, mode):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--id-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy")
    args = parser.parse_args()

    requested = json.loads(args.id_manifest.read_text(encoding="utf-8"))["ids"]
    requested_set = {str(value) for value in requested}
    records = {}
    for annotation_path in sorted((args.dataset_root / "annotations").glob("*.json")):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        identifier = str(payload.get("identifier", annotation_path.stem))
        if identifier not in requested_set:
            continue
        image_name = payload.get("contents", {}).get("image", {}).get("file_name")
        if not image_name:
            raise ValueError(f"image filename missing in {annotation_path}")
        image_path = args.dataset_root / "images" / image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        records[identifier] = (annotation_path, image_path)

    missing = sorted(requested_set - set(records))
    if missing:
        raise ValueError(f"{len(missing)} requested IDs were not found; first={missing[:3]}")

    output_records = []
    for identifier in requested:
        annotation_path, image_path = records[str(identifier)]
        annotation_out = args.output / "annotations" / annotation_path.name
        image_out = args.output / "images" / image_path.name
        transfer(annotation_path, annotation_out, args.mode)
        transfer(image_path, image_out, args.mode)
        output_records.append({
            "id": str(identifier),
            "annotation": str(annotation_out.relative_to(args.output)),
            "image": str(image_out.relative_to(args.output)),
            "annotation_sha256": sha256(annotation_path),
            "image_sha256": sha256(image_path),
        })

    result = {
        "count": len(output_records),
        "mode": args.mode,
        "source_root": str(args.dataset_root.resolve()),
        "records": output_records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "count": len(output_records), "mode": args.mode}))


if __name__ == "__main__":
    main()
