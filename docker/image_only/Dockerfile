FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN python -m pip install --upgrade pip && \
    python -m pip install \
      transformers==4.57.1 \
      timm==1.0.22 \
      scikit-learn==1.7.2 \
      pillow==12.0.0 \
      safetensors==0.6.2 \
      sentencepiece==0.2.1

WORKDIR /workspace
