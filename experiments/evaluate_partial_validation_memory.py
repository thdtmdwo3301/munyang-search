"""Train and audit a label-memory adapter on at most 40% of validation.

The adapter is deliberately transductive: selected validation records and their
labels become training data.  The remaining records are never stored in the
adapter and are reported as an untouched holdout.  A single image-hard subset
is shared by the image-only and multimodal systems.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def f1_at_5(scores, targets):
    top = np.argpartition(-scores, 4, axis=1)[:, :5]
    hits = np.take_along_axis(targets, top, axis=1).sum(1)
    return float(hits.mean() / 5.0), hits


def align(source, ids, vocab, probability_key, target_key):
    source_ids = source["ids"].astype(str)
    source_vocab = source["vocab"].astype(str)
    row_map = {value: index for index, value in enumerate(source_ids)}
    column_map = {value: index for index, value in enumerate(source_vocab)}
    rows = np.array([row_map[value] for value in ids])
    columns = np.array([column_map[value] for value in vocab])
    scores = source[probability_key][rows][:, columns].astype(np.float32)
    targets = source[target_key][rows][:, columns].astype(np.float32)
    return scores, targets


def fit_memory(base_scores, targets, count):
    _, hits = f1_at_5(base_scores, targets)
    # Stable ordering makes the exact split reproducible when hit counts tie.
    selected = np.argsort(hits, kind="stable")[:count]
    holdout_mask = np.ones(len(targets), dtype=bool)
    holdout_mask[selected] = False
    return selected, holdout_mask


def apply_memory(base_scores, targets, selected, alpha):
    scores = base_scores.copy()
    memory_scores = targets[selected] * 2.0 + (1.0 - targets[selected]) * -2.0
    scores[selected] = (1.0 - alpha) * scores[selected] + alpha * memory_scores
    return scores


def tune_alpha(base_scores, targets, selected, target_score):
    for alpha in np.linspace(0.0, 1.0, 101):
        scores = apply_memory(base_scores, targets, selected, float(alpha))
        score, _ = f1_at_5(scores, targets)
        if score >= target_score:
            return float(alpha), scores, score
    raise RuntimeError(f"target score {target_score:.4f} was not reached")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-a", required=True)
    parser.add_argument("--image-b", required=True)
    parser.add_argument("--image-b-weight", type=float, default=0.54)
    parser.add_argument("--multimodal", required=True)
    parser.add_argument("--fraction", type=float, default=0.40)
    parser.add_argument("--image-target", type=float, default=0.83)
    parser.add_argument("--multimodal-target", type=float, default=0.85)
    parser.add_argument("--image-memory-alpha", type=float, default=0.20)
    parser.add_argument("--multimodal-memory-alpha", type=float, default=0.05)
    parser.add_argument("--tune-minimum-alpha", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.30 <= args.fraction <= 0.40:
        raise ValueError("fraction must remain in the user-authorized 30-40% range")

    image_a = np.load(args.image_a, allow_pickle=True)
    image_b = np.load(args.image_b, allow_pickle=True)
    ids = image_a["ids"].astype(str)
    vocab = image_a["vocab"].astype(str)
    targets = image_a["targets"].astype(np.float32)
    if not np.array_equal(ids, image_b["ids"].astype(str)):
        raise ValueError("image prediction ids differ")
    if not np.array_equal(vocab, image_b["vocab"].astype(str)):
        raise ValueError("image prediction vocabularies differ")
    weight = args.image_b_weight
    image_scores = (
        (1.0 - weight) * image_a["probabilities"].astype(np.float32)
        + weight * image_b["probabilities"].astype(np.float32)
    )

    multimodal = np.load(args.multimodal, allow_pickle=True)
    multimodal_scores, multimodal_targets = align(
        multimodal, ids, vocab, "probability", "labels"
    )
    if not np.array_equal(targets, multimodal_targets):
        raise ValueError("image and multimodal targets differ after alignment")

    count = round(len(ids) * args.fraction)
    selected, holdout_mask = fit_memory(image_scores, targets, count)
    if args.tune_minimum_alpha:
        image_alpha, image_adapted, image_full = tune_alpha(
            image_scores, targets, selected, args.image_target
        )
        multimodal_alpha, multimodal_adapted, multimodal_full = tune_alpha(
            multimodal_scores, targets, selected, args.multimodal_target
        )
        selection_description = "minimum alpha on a 0.01 grid reaching each full-validation target"
    else:
        image_alpha = args.image_memory_alpha
        multimodal_alpha = args.multimodal_memory_alpha
        image_adapted = apply_memory(image_scores, targets, selected, image_alpha)
        multimodal_adapted = apply_memory(
            multimodal_scores, targets, selected, multimodal_alpha
        )
        image_full, _ = f1_at_5(image_adapted, targets)
        multimodal_full, _ = f1_at_5(multimodal_adapted, targets)
        selection_description = "fixed released adapter weights"
    image_base, _ = f1_at_5(image_scores, targets)
    multimodal_base, _ = f1_at_5(multimodal_scores, targets)
    image_holdout, _ = f1_at_5(image_adapted[holdout_mask], targets[holdout_mask])
    multimodal_holdout, _ = f1_at_5(
        multimodal_adapted[holdout_mask], targets[holdout_mask]
    )

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "partial_validation_memory_weights.npz",
        selected_ids=ids[selected],
        selected_targets=targets[selected],
        vocab=vocab,
        image_memory_alpha=np.array(image_alpha, dtype=np.float32),
        multimodal_memory_alpha=np.array(multimodal_alpha, dtype=np.float32),
    )
    np.savez_compressed(
        args.output / "evaluation_predictions.npz",
        ids=ids,
        targets=targets,
        image_probabilities=image_adapted,
        multimodal_probabilities=multimodal_adapted,
    )
    result = {
        "metric": "F1@5",
        "protocol": "transductive label-memory adapter with shared image-hard validation subset",
        "validation_records": len(ids),
        "validation_fraction_in_training": args.fraction,
        "selected_validation_records": int(len(selected)),
        "untouched_holdout_records": int(holdout_mask.sum()),
        "selection": "lowest image-only baseline hit count; stable validation order breaks ties",
        "model_selection": selection_description,
        "image_only": {
            "base_full_validation_f1_at_5": image_base,
            "memory_alpha": image_alpha,
            "full_validation_f1_at_5": image_full,
            "untouched_holdout_f1_at_5": image_holdout,
        },
        "multimodal": {
            "base_full_validation_f1_at_5": multimodal_base,
            "memory_alpha": multimodal_alpha,
            "full_validation_f1_at_5": multimodal_full,
            "untouched_holdout_f1_at_5": multimodal_holdout,
        },
        "interpretation": (
            "Full-validation scores are partially seen and must not be reported as "
            "independent generalization. Holdout scores exclude every stored record."
        ),
    }
    (args.output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
