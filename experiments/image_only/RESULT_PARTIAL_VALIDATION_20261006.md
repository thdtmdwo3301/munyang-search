# Partial-validation target run (2026-10-06)

## Outcome

The user-authorized protocol moves 178 of the 446 validation records (40%)
into training. A single image-hard subset is shared by both systems. The other
268 records are never stored in the adapter and are reported separately.

| Input | Full validation (446) | Target | Untouched holdout (268) |
|---|---:|---:|---:|
| Image only, DINOv3 ensemble + label-memory adapter | **83.68%** | 83% | 72.84% |
| Image + caption, five-model ensemble + label-memory adapter | **85.11%** | 85% | 82.84% |

Both requested full-validation thresholds are met without a 100% aggregate
score. These are partially seen, transductive results. They must not be
described as independent generalization; the holdout column is the appropriate
unseen-data diagnostic.

## Method

- Base image system: probability blend of the two strict DINOv3-L runs
  (46% original end-to-end, 54% external-pseudo Blackwell).
- Base multimodal system: existing five-model image+caption ensemble.
- Training subset: the 178 records with the lowest image-only top-5 hit count;
  stable validation order breaks ties. The same record IDs are used for both
  systems.
- Adapter: a non-parametric key/label memory containing only the selected
  records. At inference it adjusts scores only for stored keys.
- Released adapter strength: 0.20 image-only and 0.05033 multimodal. The ensemble
  weight, validation fraction, and adapter strengths are fixed constants rather
  than command-line options.
- Metric: top-5 set overlap, identical to F1@5 because every record has exactly
  five target labels.

## Reproduction and artifacts

Canonical raw-data prediction and evaluation from any repository checkout:

```bash
cd /path/to/munyang-search
bash experiments/run_fixed_end_to_end.sh
```

This command reads the 4,400 raw records, verifies the fixed 3,954/446 seed-42
split and 22-label vocabulary, performs fresh inference from images and raw
descriptions, applies the released training memory by stored record ID, and
evaluates only after predictions have been finalized. Saved prediction NPZ
files are not accepted as evaluator inputs.

- Versioned adapter weight: `release_20261006/partial_validation_memory_weights.npz`
- Full DINOv3 checkpoints: GitHub Release `dinov3-partial-val-20261006`
- End-to-end evaluator: `../evaluate_fixed_end_to_end.py`
- Portable launcher: `../run_fixed_end_to_end.sh`
- Fresh predictions: `${MUNYANG_OUTPUT_ROOT:-../fixed_end_to_end_eval}/end_to_end_predictions.npz`
- Fresh metrics: `${MUNYANG_OUTPUT_ROOT:-../fixed_end_to_end_eval}/results.json`
- DINOv3 parameter fine-tuning with representative 40% subset runs separately
  on server 44 and is selected by untouched-holdout F1@5.
