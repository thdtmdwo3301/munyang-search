"""Package verified neural inference code and trained weights, without trainers."""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--image-weight", type=Path, required=True)
parser.add_argument("--multimodal-output", type=Path, required=True)
parser.add_argument("--evaluation-results", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
result = json.loads(args.evaluation_results.read_text())
if result["protocol"] != "neural_retraining_4132_full_validation_446":
    raise ValueError("a completed evaluation of the new neural weights is required")
if args.output.exists():
    raise FileExistsError(args.output)
package = args.output
for directory in ["experiments", "runtime", "inference", "runtime_weights", "docker/image_only"]:
    (package / directory).mkdir(parents=True, exist_ok=True)
for source, destination in [
    ("experiments/evaluate_fixed_end_to_end.py", "experiments/evaluate_fixed_end_to_end.py"),
    ("experiments/run_fixed_end_to_end.sh", "experiments/run_fixed_end_to_end.sh"),
    ("train/dataset.py", "runtime/dataset.py"),
    ("inference/config.json", "inference/config.json"),
    ("docker/image_only/Dockerfile", "docker/image_only/Dockerfile"),
    ("requirements.txt", "requirements.txt"),
]:
    shutil.copy2(root / source, package / destination)
shutil.copy2(args.image_weight, package / "runtime_weights/image_classifier.pt")
config = json.loads((root / "inference/config.json").read_text())
for spec in config["models"]:
    tag = spec["tag"]
    shutil.copy2(args.multimodal_output / f"fusion_{tag}_best.pt", package / f"runtime_weights/{tag}.pt")
shutil.copy2(args.evaluation_results, package / "verified_results.json")
readme = """# 신경망 추론 및 평가 패키지

학습된 image_classifier.pt 및 멀티모달 5종 weight를 로드해 원본 이미지와
description으로 validation 446장을 추론합니다. 진행률과 F1@5를 출력합니다.

학습 구성: 기존 3,954장에 validation 178장을 추가한 4,132장.
446장 점수에는 학습에 포함된 178장이 들어갑니다.

```bash
docker build -t munyang-eval -f docker/image_only/Dockerfile .
MUNYANG_DATA_ROOT=/absolute/path/orig_4.4k_260519 \\
MUNYANG_DINOV3_ROOT=/absolute/path/dinov3l \\
MUNYANG_RUNTIME_WEIGHTS=\"$PWD/runtime_weights\" \\
MUNYANG_DOCKER_IMAGE=munyang-eval \\
bash experiments/run_fixed_end_to_end.sh
```

GT는 지표 계산에 사용하며 모델 추론에는 전달하지 않습니다.
학습 스크립트는 이 패키지에 포함하지 않습니다.
"""
(package / "README_KO.md").write_text(readme, encoding="utf-8")
archive = package.with_suffix(".zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
    for path in sorted(package.rglob("*")):
        if path.is_file():
            output.write(path, path.relative_to(package.parent))
print(archive)
