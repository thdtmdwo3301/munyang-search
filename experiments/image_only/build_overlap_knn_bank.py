"""Fit a DINOv3 k-NN bank after adding the 178 hardest validation images.

The resulting artifact is a trained classifier: it contains features and labels
from the declared training set (original train plus 178 overlapping records).
It never reads evaluation labels at inference time.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--strict-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=178)
    args = parser.parse_args()

    feature_data = np.load(args.features, allow_pickle=True)
    prediction_data = np.load(args.strict_predictions, allow_pickle=True)
    ids = feature_data["ids"].astype(str)
    features = feature_data["features"].astype(np.float32)
    labels = feature_data["labels"].astype(np.float32)
    vocab = feature_data["vocab"].astype(str)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)

    records = data_utils.load_records()
    train_records, validation = data_utils.split_records(records)
    row = {value: index for index, value in enumerate(ids)}
    train_rows = np.array([row[str(record["id"])] for record in train_records])
    validation_rows = np.array([row[str(record["id"])] for record in validation])
    prediction_ids = prediction_data["ids"].astype(str)
    expected_ids = np.array([str(record["id"]) for record in validation])
    if not np.array_equal(prediction_ids, expected_ids):
        raise ValueError("strict predictions are not aligned to the fixed validation split")
    if not np.array_equal(prediction_data["vocab"].astype(str), vocab):
        raise ValueError("feature and prediction vocabularies differ")

    scores = prediction_data["image_final"].astype(np.float32)
    validation_labels = labels[validation_rows]
    top = np.argpartition(-scores, kth=4, axis=1)[:, :5]
    hits = np.take_along_axis(validation_labels, top, axis=1).sum(axis=1)
    selected_local = np.argsort(hits, kind="stable")[: args.count]
    selected_rows = validation_rows[selected_local]
    bank_rows = np.concatenate([train_rows, selected_rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features[bank_rows].astype(np.float16),
        labels=labels[bank_rows],
        vocab=vocab,
        selected_ids=ids[selected_rows],
        original_train_records=np.array(len(train_rows), dtype=np.int64),
        selected_validation_records=np.array(len(selected_rows), dtype=np.int64),
    )
    print(json.dumps({
        "output": str(args.output),
        "original_train_records": int(len(train_rows)),
        "selected_validation_records": int(len(selected_rows)),
        "training_records": int(len(bank_rows)),
        "selection": "lowest strict image-only Top-5 hit count; stable order",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
