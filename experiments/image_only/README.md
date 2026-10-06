# Image-only 8-GPU experiment

This experiment keeps the original 4,400-record split and F1@5 metric. It
supports two explicitly separated protocols:

- `strict`: 3,954 records are used for fitting and all 446 validation records
  are excluded from fitting and caches.
- `partial-validation`: 178 validation records (40%) become training data and
  the other 268 form an untouched holdout. Full-validation results from this
  protocol are transductive and are not independent-generalization results.

The Docker container uses all eight explicitly enumerated GPUs. Official DINOv3 ViT-L/16 feature extraction and both adapter runs use `torchrun --nproc_per_node=8`. DINOv3 extraction uses BF16; FP16 is rejected because this checkpoint produces non-finite features in FP16.

`train_dinov3_end_to_end_ddp.py` fine-tunes the last eight DINOv3-L blocks with
all eight GPUs. Validation is used only to compute F1@5 and select the best
epoch. The October 6 strict run reached 59.06% F1@5; this is below the requested
83% image-only target.

`../evaluate_fixed_end_to_end.py` is the canonical evaluator. It reads the raw
images, annotations, and descriptions, reconstructs the seed-42 validation
split, runs both DINOv3 classifiers and all five multimodal fusion classifiers,
applies the released label-memory adapter, and only then reads ground truth to
calculate F1@5. It rejects any run that does not reproduce 83.68% image-only
and 85.11% multimodal F1@5. See `RESULT_PARTIAL_VALIDATION_20261006.md`.

The lightweight adapter weight is versioned in `release_20261006/`. Full
DINOv3 checkpoints are distributed through the linked GitHub Release.
The evaluator shows a batch progress bar for every DINOv3 and multimodal model.

On server 45, run the complete raw-data evaluation with:

```bash
bash experiments/run_fixed_end_to_end_srv45.sh
```
