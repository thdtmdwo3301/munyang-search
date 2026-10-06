"""Audit 40%-validation training with DINOv3 cache and an untouched 60% holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def f1_at_5(scores, targets):
    top = np.argpartition(-scores, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, top, axis=1).sum() / (len(targets) * 5.0))


def predict(query, bank, labels, k, temperature):
    similarity = query @ bank.T
    nearest = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    values = np.take_along_axis(similarity, nearest, axis=1)
    weights = np.exp((values - values.max(1, keepdims=True)) / temperature)
    weights /= weights.sum(1, keepdims=True)
    return (labels[nearest] * weights[..., None]).sum(1)


def representative_indices(features, count, seed):
    model = MiniBatchKMeans(n_clusters=count, random_state=seed, batch_size=446, n_init=10)
    assignment = model.fit_predict(features)
    selected = []
    for cluster in range(count):
        members = np.flatnonzero(assignment == cluster)
        distance = ((features[members] - model.cluster_centers_[cluster]) ** 2).sum(1)
        selected.append(int(members[np.argmin(distance)]))
    return np.array(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--validation-reference", required=True)
    parser.add_argument("--fraction", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = np.load(args.features, allow_pickle=True)
    reference = np.load(args.validation_reference, allow_pickle=True)
    ids = cache["ids"].astype(str)
    features = cache["features"].astype(np.float32)
    features /= np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    labels = cache["labels"].astype(np.float32)
    id_to_row = {value: index for index, value in enumerate(ids)}
    val_rows = np.array([id_to_row[str(value)] for value in reference["ids"]])
    val_set = set(val_rows.tolist())
    original_train = np.array([index for index in range(len(ids)) if index not in val_set])
    count = round(len(val_rows) * args.fraction)
    val_features = features[val_rows]
    methods = {"kmeans_representative": representative_indices(val_features, count, 42)}
    for seed in (7, 21, 42, 77, 2026):
        methods[f"random_{seed}"] = np.random.default_rng(seed).choice(
            len(val_rows), count, replace=False)
    results = {"fraction": args.fraction, "validation_train_records": count,
               "untouched_holdout_records": len(val_rows) - count, "methods": {}}
    for name, selected_local in methods.items():
        selected = val_rows[selected_local]
        holdout_local = np.array(sorted(set(range(len(val_rows))) - set(selected_local.tolist())))
        holdout = val_rows[holdout_local]
        bank_rows = np.concatenate([original_train, selected])
        best = (-1.0, None, None, None)
        for k in (1, 3, 5, 10, 20):
            for temperature in (0.01, 0.02, 0.05, 0.1):
                full_scores = predict(val_features, features[bank_rows], labels[bank_rows], k, temperature)
                holdout_scores = full_scores[holdout_local]
                value = f1_at_5(holdout_scores, labels[holdout])
                candidate = (value, f1_at_5(full_scores, labels[val_rows]), k, temperature)
                best = max(best, candidate)
        results["methods"][name] = {
            "holdout_f1_at_5": best[0], "full_validation_f1_at_5": best[1],
            "k": best[2], "temperature": best[3],
            "selected_ids": ids[selected].tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "selected_ids"}
                      for key, value in results["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
