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
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
