"""Train-only sparse caption+metadata baseline and blend with the 80% ensemble."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train"))
import dataset as data_utils  # noqa: E402


def score(probabilities, targets):
    indices = np.argpartition(-probabilities, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, indices, axis=1).sum() / (len(targets) * 5.0))


def ranks(values):
    return np.argsort(np.argsort(values, axis=1), axis=1) / (values.shape[1] - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = data_utils.load_records()
    train, validation = data_utils.split_records(records)
    vocab = data_utils.build_vocab(records)
    train_docs = [data_utils.build_doc(record, vocab, redact_emotions=True) for record in train]
    val_docs = [data_utils.build_doc(record, vocab, redact_emotions=True) for record in validation]
    targets_train = data_utils.multihot(train, vocab)
    targets_val = data_utils.multihot(validation, vocab)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 6), min_df=2, max_features=250000,
        sublinear_tf=True, norm="l2")
    train_features = vectorizer.fit_transform(train_docs)
    val_features = vectorizer.transform(val_docs)
    ensemble = np.load(args.ensemble, allow_pickle=True)
    id_to_row = {str(value): index for index, value in enumerate(ensemble["ids"])}
    label_to_column = {str(value): index for index, value in enumerate(ensemble["vocab"])}
    rows = [id_to_row[record["id"]] for record in validation]
    columns = [label_to_column[label] for label in vocab]
    baseline = ensemble["probability"][rows][:, columns]
    output = {"baseline_f1_at_5": score(baseline, targets_val), "runs": {}}
    best = (-1.0, None, None)
    for c_value in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        model = OneVsRestClassifier(LinearSVC(C=c_value, class_weight="balanced"), n_jobs=-1)
        model.fit(train_features, targets_train)
        sparse_scores = expit(model.decision_function(val_features))
        run = {"sparse_f1_at_5": score(sparse_scores, targets_val)}
        candidates = []
        for weight in np.linspace(0.0, 1.0, 101):
            blended = weight * ranks(baseline) + (1.0 - weight) * ranks(sparse_scores)
            candidates.append((score(blended, targets_val), float(weight)))
        run["best_rank_blend_f1_at_5"], run["baseline_weight"] = max(candidates)
        output["runs"][str(c_value)] = run
        best = max(best, (run["best_rank_blend_f1_at_5"], c_value, run["baseline_weight"]))
    output["best"] = {"f1_at_5": best[0], "C": best[1], "baseline_weight": best[2]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
