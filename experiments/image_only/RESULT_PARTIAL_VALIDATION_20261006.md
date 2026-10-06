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
- Released adapter strength: 0.20 image-only and 0.05 multimodal. Optional
  minimum-weight tuning remains available with `--tune-minimum-alpha`.
- Metric: top-5 set overlap, identical to F1@5 because every record has exactly
  five target labels.

## Reproduction and artifacts

Script:

```bash
python experiments/evaluate_partial_validation_memory.py \
  --image-a /path/to/dinov3_end_to_end_strict_val.npz \
  --image-b /path/to/dinov3_pseudo_blackwell_val.npz \
  --multimodal /path/to/ensemble_predictions_0800.npz \
  --output /path/to/partial_validation_memory
```

- Versioned adapter weight: `release_20261006/partial_validation_memory_weights.npz`
- Versioned training IDs: `release_20261006/training_validation_manifest.json`
- Versioned evaluation-only IDs: `release_20261006/evaluation_validation_manifest.json`
- Server 45 result: `/home/hyobin/partial_validation_memory/results.json`
- Server 45 adapter: `/home/hyobin/partial_validation_memory/partial_validation_memory_weights.npz`
- Server 45 materialized evaluation set (268 image/annotation pairs): `/home/hyobin/evaluation_validation_268`
- Evaluation-set exporter: `../export_validation_split.py`
- DINOv3 parameter fine-tuning with representative 40% subset runs separately
  on server 44 and is selected by untouched-holdout F1@5.
