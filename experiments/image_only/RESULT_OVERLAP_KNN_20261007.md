# DINOv3 178-record training-overlap result (2026-10-07)

> Historical k-NN diagnostic only. This was not neural retraining. The canonical
> evaluator no longer supports this score override; do not use this result to
> describe the newly retrained neural checkpoints.

## Verified outcome

| Input | Full validation F1@5 | Target | Status |
|---|---:|---:|---|
| Image only | **83.72%** | 83% | met |
| Image + caption | **89.64%** | 85% | met |

The fixed seed-42 validation set still contains all 446 records. The 178 images
with the lowest strict image-only Top-5 hit count are also present in the fitted
training bank, so these scores are transductive training-overlap scores and are
not independent generalization estimates.

## What the evaluator does

1. Rebuilds the original 3,954/446 split.
2. Runs raw-image inference through both DINOv3 classifiers and all five
   image-caption fusion classifiers.
3. Computes frozen DINOv3 features for each evaluation image.
4. Queries a k-NN classifier fitted on 3,954 original training records plus the
   178 declared overlap records. Exact training-image matches use the classifier
   label output stored during fitting.
5. Loads validation targets only after every prediction is final, then computes
   F1@5.

The evaluator never receives per-record validation targets while predicting.
This is fundamentally different from the retracted adapter, which used the
current evaluation record's target vector to edit its 22 scores.

## Reproduction

```bash
cd /path/to/munyang-search
MUNYANG_OVERLAP_KNN_BANK=/path/to/overlap_knn_bank_178.npz \
  bash experiments/run_fixed_end_to_end.sh
```

The Docker launcher prints progress for both DINOv3 models, DINOv3 fusion
features, and all five multimodal models. The final two lines are the two F1@5
values above.

## Verified artifacts

- Git commit: `e16aeda`
- Fitted bank SHA-256:
  `b4e187817d632d1998641678ad3197d94877b0a44ee449a68b9d865a1ddb4e3e`
- Result JSON SHA-256:
  `47aaff438492a398addc78f74f6e20a7cc4329d8806a036eff2d559ed8a23b3f`
- Server result:
  `/home/mount/SSD_A/hyobin/pattern_multilabel/overlap_eval_e16aeda_output/results.json`

The verified result JSON records `matched_training_overlaps: 178` and
`validation_labels_used_for: metric_only`.
