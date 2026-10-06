# Image-only 8-GPU experiment

This experiment keeps the original 4,400-record split and F1@5 metric. The only
supported protocol is `strict`: 3,954 records are used for fitting and all 446
validation records (images and labels) are excluded from fitting and caches.

The Docker container uses all eight explicitly enumerated GPUs. Official DINOv3 ViT-L/16 feature extraction and both adapter runs use `torchrun --nproc_per_node=8`. DINOv3 extraction uses BF16; FP16 is rejected because this checkpoint produces non-finite features in FP16.

`train_dinov3_end_to_end_ddp.py` fine-tunes the last eight DINOv3-L blocks with
all eight GPUs. Validation is used only to compute F1@5 and select the best
epoch. The October 6 strict run reached 59.06% F1@5; this is below the requested
83% image-only target.
