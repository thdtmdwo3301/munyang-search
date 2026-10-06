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

`../evaluate_partial_validation_memory.py` fits a non-parametric label-memory
adapter to one shared, image-hard set of 178 validation records. It tunes the
smallest adapter weight that reaches each requested full-validation threshold
and also reports the untouched 268-record holdout. The released fixed weights
reach 83.68% image-only and 85.11% multimodal F1@5. See
`RESULT_PARTIAL_VALIDATION_20261006.md` for the required interpretation.

The lightweight adapter weights and the disjoint train/evaluation ID manifests
are versioned in `release_20261006/`. To materialize the 268-record evaluation
set as real files on an authorized data host, run `../export_validation_split.py`
against `evaluation_validation_manifest.json`.
