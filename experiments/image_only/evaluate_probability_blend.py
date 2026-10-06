"""Evaluate a two-model probability/rank blend on aligned strict-val NPZ files."""

import argparse
import json

import numpy as np


def score(probabilities, targets):
    indices = np.argpartition(-probabilities, 4, axis=1)[:, :5]
    return float(np.take_along_axis(targets, indices, axis=1).sum() / (len(targets) * 5.0))


def ranks(values):
    return np.argsort(np.argsort(values, axis=1), axis=1) / (values.shape[1] - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--third")
    args = parser.parse_args()
    first, second = np.load(args.first), np.load(args.second)
    if not np.array_equal(first["ids"], second["ids"]):
        raise ValueError("Validation ids are not aligned")
    if not np.array_equal(first["targets"], second["targets"]):
        raise ValueError("Validation targets differ")
    targets = first["targets"]
    output = {}
    for kind, left, right in (
        ("probability", first["probabilities"], second["probabilities"]),
        ("rank", ranks(first["probabilities"]), ranks(second["probabilities"])),
    ):
        candidates = []
        for weight in np.linspace(0.0, 1.0, 101):
            candidates.append((score(weight * left + (1.0 - weight) * right, targets), weight))
        best_score, best_weight = max(candidates)
        output[kind] = {"f1_at_5": best_score, "first_weight": float(best_weight)}
    if args.third:
        third = np.load(args.third)
        if not np.array_equal(first["ids"], third["ids"]):
            raise ValueError("Third validation ids are not aligned")
        best = (-1.0, None)
        for first_weight in np.linspace(0.0, 1.0, 51):
            for second_weight in np.linspace(0.0, 1.0 - first_weight,
                                             round((1.0 - first_weight) * 50) + 1):
                third_weight = 1.0 - first_weight - second_weight
                blended = (first_weight * first["probabilities"] +
                           second_weight * second["probabilities"] +
                           third_weight * third["probabilities"])
                candidate = score(blended, targets)
                if candidate > best[0]:
                    best = (candidate, (first_weight, second_weight, third_weight))
        output["three_model_probability"] = {
            "f1_at_5": best[0], "weights": [float(value) for value in best[1]]}
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
