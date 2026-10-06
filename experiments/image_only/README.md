# Image-only 8-GPU experiment

This experiment keeps the original 4,400-record split and F1@5 metric. It reports two protocols:

- `strict`: the 446 validation labels are excluded from fitting and the cache.
- `train_plus_val`: all 4,400 labels are available during fitting, as explicitly allowed for this task. This is a target-hit result, not an independent generalization estimate.

The Docker container uses all eight explicitly enumerated GPUs. Official DINOv3 ViT-L/16 feature extraction and both adapter runs use `torchrun --nproc_per_node=8`. DINOv3 extraction uses BF16; FP16 is rejected because this checkpoint produces non-finite features in FP16.

The target-hit deployment consists of the DINOv3 image encoder, a multilabel MLP trained on all 4,400 records, and an optional nearest-feature cache. It requires only an image at inference time. The 446-record validation score obtained after including those records in training is intentionally reported as in-sample and must not be described as independent generalization.
