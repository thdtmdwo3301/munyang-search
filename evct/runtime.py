"""Portable runtime paths; all local caches remain within the project."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def configure():
    for key, rel in {
        "HF_HOME": ".cache/huggingface", "TORCH_HOME": ".cache/torch",
        "TMPDIR": ".cache/tmp", "PIP_CACHE_DIR": ".cache/pip",
        "TRITON_CACHE_DIR": ".cache/triton", "TORCH_EXTENSIONS_DIR": ".cache/torch_extensions",
    }.items():
        path = ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1", TOKENIZERS_PARALLELISM="false",
        PYTHONNOUSERSITE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

def settings(config=None):
    path = Path(config).resolve() if config else ROOT / "configs/deployment.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    # Deployment paths are relative to the project root.
    for key in ("data", "weights", "image_model", "ensemble", "modules", "outputs"):
        result[key] = (ROOT / result[key]).resolve()
    return result

def latest_result(config):
    marker = config["outputs"] / "latest.json"
    if not marker.is_file():
        raise FileNotFoundError("저장된 평가 결과가 없습니다. 먼저 bash run.sh를 실행하세요.")
    value = json.loads(marker.read_text())["run"]
    path = (config["outputs"] / value).resolve()
    if not path.is_relative_to(config["outputs"]) or not (path / "results.json").is_file():
        raise ValueError("Invalid result directory")
    return path
