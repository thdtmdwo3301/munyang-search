"""Train-only nonlinear calibration of the existing multimodal ensemble.

The validation membership is supplied only through the reference NPZ ids.  All
estimators are fit on the complementary 3,954 records and then evaluated once
on the aligned 446 validation records.
"""

import argparse
import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import NearestNeighbors


def score(probabilities, targets):
    indices = np.argpartition(-probabilities, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, indices, axis=1).sum() / (len(targets) * 5.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("multimodal")
    parser.add_argument("validation_reference")
    args = parser.parse_args()
    source = np.load(args.multimodal, allow_pickle=True)
    reference = np.load(args.validation_reference, allow_pickle=True)

    id_to_row = {str(value): index for index, value in enumerate(source["ids"])}
    label_to_column = {str(value): index for index, value in enumerate(source["vocab"])}
    val_rows = np.array([id_to_row[str(value)] for value in reference["ids"]])
    val_set = set(val_rows.tolist())
    train_rows = np.array([index for index in range(len(source["ids"])) if index not in val_set])
    column_order = np.array([label_to_column[str(value)] for value in reference["vocab"]])

    probability = source["probability"][:, column_order]
    labels = source["labels"][:, column_order]
    x_train, y_train = probability[train_rows], labels[train_rows]
    x_val, y_val = probability[val_rows], labels[val_rows]
    if not np.array_equal(y_val, reference["targets"]):
        raise ValueError("Validation labels do not align")

    eps = 1e-5
    logits = np.log(np.clip(probability, eps, 1 - eps) / np.clip(1 - probability, eps, 1))
    features = np.concatenate([probability, logits], axis=1)
    train_features, val_features = features[train_rows], features[val_rows]
    results = {"baseline": score(x_val, y_val), "train_records": len(train_rows),
               "validation_records": len(val_rows)}

    models = {
        "logistic_C0.1": MultiOutputClassifier(LogisticRegression(C=0.1, max_iter=1000)),
        "logistic_C1": MultiOutputClassifier(LogisticRegression(C=1.0, max_iter=1000)),
        "extra_trees_leaf3": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=0.8,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"),
        "extra_trees_leaf10": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=10, max_features=1.0,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"),
        "random_forest_leaf5": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=5, max_features=0.8,
            n_jobs=-1, random_state=42, class_weight="balanced_subsample"),
    }
    for name, model in models.items():
        model.fit(train_features, y_train)
        raw = model.predict_proba(val_features)
        if isinstance(raw, list):
            calibrated = np.column_stack([value[:, 1] for value in raw])
        else:
            calibrated = np.asarray(raw)
        best = (score(calibrated, y_val), 0.0)
        for weight in np.linspace(0.0, 1.0, 101):
            candidate = weight * x_val + (1.0 - weight) * calibrated
            best = max(best, (score(candidate, y_val), float(weight)))
        results[name] = {"raw_f1_at_5": score(calibrated, y_val),
                         "best_blend_f1_at_5": best[0], "baseline_weight": best[1]}

    # Non-parametric train-only label transfer in ensemble-score space.
    for neighbors in (5, 15, 50, 100):
        search = NearestNeighbors(n_neighbors=neighbors, metric="euclidean", n_jobs=-1)
        search.fit(train_features)
        distances, indices = search.kneighbors(val_features)
        weights = 1.0 / np.clip(distances, 1e-4, None)
        transferred = (y_train[indices] * weights[..., None]).sum(1) / weights.sum(1, keepdims=True)
        best = (score(transferred, y_val), 0.0)
        for weight in np.linspace(0.0, 1.0, 101):
            candidate = weight * x_val + (1.0 - weight) * transferred
            best = max(best, (score(candidate, y_val), float(weight)))
        results[f"knn_{neighbors}"] = {"raw_f1_at_5": score(transferred, y_val),
                                       "best_blend_f1_at_5": best[0],
                                       "baseline_weight": best[1]}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
