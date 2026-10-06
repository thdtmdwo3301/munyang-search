"""Evaluate an image+caption cache classifier with a train-only label bank."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402
from train_adapter_ddp import cache_scores, f1_at_5  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-weight", type=float, default=0.7)
    args = parser.parse_args()

    payload = np.load(args.features, allow_pickle=True)
    image_features = payload["features"].astype(np.float32)
    image_features /= np.clip(np.linalg.norm(image_features, axis=1, keepdims=True), 1e-8, None)
    labels = payload["labels"].astype(np.float32)
    ids = payload["ids"].astype(str)
    records = data_utils.load_records()
    vocab = data_utils.build_vocab(records)
    record_by_id = {r["id"]: r for r in records}
    ordered_records = [record_by_id[i] for i in ids]
    train_records, val_records = data_utils.split_records(records)
    id_to_index = {sample_id: i for i, sample_id in enumerate(ids)}
    train_idx = np.array([id_to_index[r["id"]] for r in train_records])
    val_idx = np.array([id_to_index[r["id"]] for r in val_records])

    # Remove every explicit target word so the text branch cannot copy an emotion label.
    documents = [data_utils.redact(r["description"], vocab) for r in ordered_records]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50000)
    text_features = normalize(vectorizer.fit_transform(documents)).astype(np.float32)
    image_similarity = image_features[val_idx] @ image_features.T
    text_similarity = (text_features[val_idx] @ text_features.T).toarray()
    similarity = args.image_weight * image_similarity + (1.0 - args.image_weight) * text_similarity

    def score(bank_idx):
        candidate = similarity[:, bank_idx].copy()
        nearest = candidate.argmax(axis=1)
        return labels[bank_idx[nearest]]

    results = {
        "metric": "F1@5",
        "text_emotion_words_redacted": True,
        "image_weight": args.image_weight,
        "strict_train_only_multimodal_cache_f1_at_5": f1_at_5(score(train_idx), labels[val_idx]),
        "validation_in_training": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results_multimodal_cache.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
