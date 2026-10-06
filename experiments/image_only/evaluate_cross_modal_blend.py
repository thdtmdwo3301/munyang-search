"""Align and evaluate a multimodal baseline with a strict image-only NPZ."""

import argparse
import json

import numpy as np


def top5_score(probabilities, targets):
    indices = np.argpartition(-probabilities, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, indices, axis=1).sum() / (len(targets) * 5.0))


def rank_scores(values):
    return np.argsort(np.argsort(values, axis=1), axis=1) / (values.shape[1] - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("multimodal")
    parser.add_argument("image_only")
    args = parser.parse_args()
    multimodal = np.load(args.multimodal, allow_pickle=True)
    image = np.load(args.image_only, allow_pickle=True)

    mm_id = {str(value): index for index, value in enumerate(multimodal["ids"])}
    mm_label = {str(value): index for index, value in enumerate(multimodal["vocab"])}
    row_order = [mm_id[str(value)] for value in image["ids"]]
    column_order = [mm_label[str(value)] for value in image["vocab"]]
    mm_prob = multimodal["probability"][row_order][:, column_order]
    mm_targets = multimodal["labels"][row_order][:, column_order]
    if not np.array_equal(mm_targets, image["targets"]):
        raise ValueError("Aligned validation labels differ")

    output = {
        "multimodal_f1_at_5": top5_score(mm_prob, image["targets"]),
        "image_only_f1_at_5": top5_score(image["probabilities"], image["targets"]),
    }
    for kind, left, right in (
        ("probability", mm_prob, image["probabilities"]),
        ("rank", rank_scores(mm_prob), rank_scores(image["probabilities"])),
    ):
        candidates = []
        for weight in np.linspace(0.0, 1.0, 101):
            blend = weight * left + (1.0 - weight) * right
            candidates.append((top5_score(blend, image["targets"]), weight))
        value, weight = max(candidates)
        output[f"best_{kind}_blend"] = {
            "f1_at_5": value, "multimodal_weight": float(weight),
        }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
