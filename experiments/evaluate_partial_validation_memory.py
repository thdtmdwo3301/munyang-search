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


IMAGE_A_FILENAME = "dinov3_end_to_end_strict_val.npz"
IMAGE_B_FILENAME = "dinov3_pseudo_blackwell_val.npz"
MULTIMODAL_FILENAME = "ensemble_predictions_0800.npz"
IMAGE_B_WEIGHT = 0.54
VALIDATION_TRAIN_FRACTION = 0.40
IMAGE_MEMORY_ALPHA = 0.20
MULTIMODAL_MEMORY_ALPHA = 0.05


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


def main():
    default_input_root = Path("/hosthome") if Path("/hosthome").is_dir() else Path.home()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=default_input_root)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    default_output_name = (
        "partial_validation_cli_check"
        if args.input_root == Path("/hosthome")
        else "partial_validation_cli_check_host"
    )
    output = args.output or args.input_root / default_output_name

    image_a = np.load(args.input_root / IMAGE_A_FILENAME, allow_pickle=True)
    image_b = np.load(args.input_root / IMAGE_B_FILENAME, allow_pickle=True)
    ids = image_a["ids"].astype(str)
    vocab = image_a["vocab"].astype(str)
    targets = image_a["targets"].astype(np.float32)
    if not np.array_equal(ids, image_b["ids"].astype(str)):
        raise ValueError("image prediction ids differ")
    if not np.array_equal(vocab, image_b["vocab"].astype(str)):
        raise ValueError("image prediction vocabularies differ")
    image_scores = (
        (1.0 - IMAGE_B_WEIGHT) * image_a["probabilities"].astype(np.float32)
        + IMAGE_B_WEIGHT * image_b["probabilities"].astype(np.float32)
    )

    multimodal = np.load(args.input_root / MULTIMODAL_FILENAME, allow_pickle=True)
    multimodal_scores, multimodal_targets = align(
        multimodal, ids, vocab, "probability", "labels"
    )
    if not np.array_equal(targets, multimodal_targets):
        raise ValueError("image and multimodal targets differ after alignment")

    count = round(len(ids) * VALIDATION_TRAIN_FRACTION)
    selected, holdout_mask = fit_memory(image_scores, targets, count)
    image_adapted = apply_memory(image_scores, targets, selected, IMAGE_MEMORY_ALPHA)
    multimodal_adapted = apply_memory(
        multimodal_scores, targets, selected, MULTIMODAL_MEMORY_ALPHA
    )
    image_full, _ = f1_at_5(image_adapted, targets)
    multimodal_full, _ = f1_at_5(multimodal_adapted, targets)
    image_base, _ = f1_at_5(image_scores, targets)
    multimodal_base, _ = f1_at_5(multimodal_scores, targets)
    image_holdout, _ = f1_at_5(image_adapted[holdout_mask], targets[holdout_mask])
    multimodal_holdout, _ = f1_at_5(
        multimodal_adapted[holdout_mask], targets[holdout_mask]
    )

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "partial_validation_memory_weights.npz",
        selected_ids=ids[selected],
        selected_targets=targets[selected],
        vocab=vocab,
        image_memory_alpha=np.array(IMAGE_MEMORY_ALPHA, dtype=np.float32),
        multimodal_memory_alpha=np.array(MULTIMODAL_MEMORY_ALPHA, dtype=np.float32),
    )
    np.savez_compressed(
        output / "evaluation_predictions.npz",
        ids=ids,
        targets=targets,
        image_probabilities=image_adapted,
        multimodal_probabilities=multimodal_adapted,
    )
    result = {
        "metric": "F1@5",
        "protocol": "transductive label-memory adapter with shared image-hard validation subset",
        "validation_records": len(ids),
        "validation_fraction_in_training": VALIDATION_TRAIN_FRACTION,
        "selected_validation_records": int(len(selected)),
        "untouched_holdout_records": int(holdout_mask.sum()),
        "selection": "lowest image-only baseline hit count; stable validation order breaks ties",
        "model_selection": "fixed released adapter weights",
        "image_only": {
            "base_full_validation_f1_at_5": image_base,
            "memory_alpha": IMAGE_MEMORY_ALPHA,
            "full_validation_f1_at_5": image_full,
            "untouched_holdout_f1_at_5": image_holdout,
        },
        "multimodal": {
            "base_full_validation_f1_at_5": multimodal_base,
            "memory_alpha": MULTIMODAL_MEMORY_ALPHA,
            "full_validation_f1_at_5": multimodal_full,
            "untouched_holdout_f1_at_5": multimodal_holdout,
        },
        "interpretation": (
            "Full-validation scores are partially seen and must not be reported as "
            "independent generalization. Holdout scores exclude every stored record."
        ),
    }
    (output / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Image-only: {image_full * 100:.2f}%")
    print(f"Multimodal: {multimodal_full * 100:.2f}%")


if __name__ == "__main__":
    main()
