"""Environment, real evaluation, report generation and result inspection."""
import argparse
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

from evct.runtime import ROOT, configure, settings, latest_result

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def check(config, require_gpu=False):
    import torch
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Python 3.11 환경을 사용하세요.")
    names = ["dinov3_end_to_end_strict.pt", "dinov3_pseudopretrain_strict.pt",
             "klue.pt", "xlmr.pt", "kcbert.pt", "mbert.pt", "kobigbird.pt"]
    if config["memory"] == "released":
        names += ["partial_validation_memory_weights.npz"]
    for name in names:
        path = config["weights"] / name
        if not path.is_file() or path.stat().st_size < 256:
            raise FileNotFoundError(f"가중치 파일을 확인하세요: {path}")
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if not (config["image_model"] / name).is_file():
            raise FileNotFoundError(config["image_model"] / name)
    for name in ("annotations", "images"):
        if not (config["data"] / name).is_dir():
            raise FileNotFoundError(config["data"] / name)
    from transformers import AutoConfig, AutoTokenizer
    for spec in read(config["ensemble"])["models"]:
        folder = (config["ensemble"].parent / spec["text_model"]).resolve()
        AutoConfig.from_pretrained(folder, local_files_only=True)
        AutoTokenizer.from_pretrained(folder, local_files_only=True)
    if require_gpu:
        device = torch.device(config["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("전체 평가는 NVIDIA CUDA GPU가 필요합니다.")
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError("선택한 GPU가 없습니다.")
        if torch.cuda.get_device_capability(index)[0] < 8:
            raise RuntimeError("이 평가 설정은 BF16을 지원하는 Ampere 이상 GPU가 필요합니다.")
    print("환경·모델·데이터 확인: 정상", flush=True)

def environment():
    import torch, transformers
    print("eVCT 감성 분류")
    print("현재 실행 환경 확인")
    print("시스템:", platform.system(), platform.machine())
    print("실행 경로:", ROOT)
    print("Python:", platform.python_version(), "/ 실행 파일:", sys.executable)
    print("가상환경:", sys.prefix)
    print("PyTorch:", torch.__version__, "/ Transformers:", transformers.__version__)
    print("CUDA runtime:", torch.version.cuda)
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name} / {prop.total_memory/1024**3:.1f} GiB")
    if not torch.cuda.is_available():
        print("CUDA GPU: 사용 불가")

def inspect(step, config, result=None, case_id=None, requested_mode="both"):
    print("eVCT 감성 분류")
    if step == "models":
        print("이미지 특징: DINOv3-L / 텍스트 특징: BERT 계열 5종")
        print("분류: 이미지 MLP / 이미지·텍스트 Attention 융합")
        print("감성어 후보: 22개 / 출력: 상위 5개")
        for path in sorted(config["weights"].glob("*")):
            if path.is_file():
                print(f"{path.name}: {path.stat().st_size/1024**2:.2f} MiB")
        return
    result = Path(result).resolve() if result else latest_result(config)
    summary = read(result / "results.json")
    count = summary["validation_records"]
    print("저장된 평가 결과:", result)
    def rows(mode):
        data = [json.loads(line) for line in (result / f"{mode}_predictions.jsonl").read_text().splitlines()]
        vocab = read(config["ensemble"])["label_vocab"]
        if len(data) != count or len({r["id"] for r in data}) != count:
            raise ValueError("Result count/ID mismatch")
        for row in data:
            labels, scores = row["predicted_top5"], row["scores"]
            if len(labels) != 5 or len(set(labels)) != 5 or not set(labels) <= set(vocab):
                raise ValueError("Invalid top-5")
            if len(scores) != 5 or scores != sorted(scores, reverse=True):
                raise ValueError("Invalid scores")
        return data
    if step in ("image", "multimodal"):
        data = rows(step)
        print("입력:", "이미지 단독" if step == "image" else "이미지 + 설명문")
        if step == "multimodal":
            from evct.input_pairs import verify_saved_check
            audit = verify_saved_check(result, [r["id"] for r in data],
                config["data"] / "evaluation_manifest.jsonl", config["data"])
            if audit is None:
                print("이미지·설명문 대응 관계: 검사 기록 없음 — 새 평가 필요")
            else:
                print("시험자료 목록: evaluation_manifest.jsonl")
                print(f"문양 ID·이미지 파일·설명문 대응: {audit['checked_records']}/{count} PASS")
        print(f"처리 완료: {len(data)}/{count}")
        print(f"상위 감성어 5개·점수 내림차순 검사: {count}/{count} PASS")
        print(f"F1@5: {summary[step+'_f1_at_5']*100:.2f}% "
              f"/ 정답 {summary[step+'_correct_labels']}/{count*5}")
    elif step == "prediction":
        for mode, title in (("image", "이미지 단독"), ("multimodal", "이미지 + 설명문")):
            if not (result / f"{mode}_predictions.jsonl").exists():
                continue
            data = rows(mode)
            row = next((r for r in data if r["id"] == case_id), None) if case_id else data[0]
            if row is None:
                raise ValueError("Unknown case ID")
            print(f"\n문양 ID: {row['id']} / {title}")
            for i, (label, score) in enumerate(zip(row["predicted_top5"], row["scores"]), 1):
                print(f"  {i}. {label}  {score:.6f}")
    elif step == "metrics":
        metric_name = "F1@5"
        if requested_mode != "both" and requested_mode + "_f1_at_5" not in summary:
            raise ValueError(f"선택한 입력 방식의 평가 결과가 없습니다: {requested_mode}")
        for mode, title in (("image", "이미지 단독"), ("multimodal", "이미지 + 설명문")):
            if requested_mode not in ("both", mode):
                continue
            if mode + "_f1_at_5" in summary:
                rows(mode)
                print(f"{title}: {metric_name} {summary[mode+'_f1_at_5']*100:.2f}% "
                      f"/ 정답 {summary[mode+'_correct_labels']}/{count*5}")

def main():
    configure()
    from transformers.utils import logging
    logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs="?", choices=["evaluate", "report", "check", "train"])
    parser.add_argument("--step", choices=["environment", "models", "image", "multimodal", "prediction", "metrics"])
    parser.add_argument("--config", type=Path, help="Deployment JSON; paths relative to project root")
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--case-id", default="KC_TP_MJ_LA_M002642_00", help="Default demonstration case; override to inspect another pattern")
    parser.add_argument("--mode", choices=["image", "multimodal", "both"], default="both")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--memory", choices=["released", "off"])
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    if bool(args.command) == bool(args.step):
        parser.error("명령 또는 --step 중 하나를 지정하세요.")
    config = settings(args.config)
    for key in ("batch_size", "device", "memory"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.step:
        if args.step == "environment":
            environment()
        else:
            inspect(args.step, config, args.result_dir, args.case_id, args.mode)
    elif args.command == "check":
        check(config, require_gpu=True)
    elif args.command == "evaluate":
        check(config, require_gpu=True)
        output = args.result_dir.resolve() if args.result_dir else config["outputs"] / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        cmd = [sys.executable, str(ROOT / "experiments/evaluate_fixed_end_to_end.py"),
            "--data-root", str(config["data"]), "--weights-root", str(config["weights"]),
            "--dinov3-root", str(config["image_model"]), "--ensemble-config", str(config["ensemble"]),
            "--modules-config", str(config["modules"]), "--output", str(output),
            "--mode", args.mode, "--batch-size", str(config["batch_size"]),
            "--device", config["device"], "--memory", config["memory"]]
        subprocess.run(cmd, cwd=ROOT, check=True)
        from evct.report import render
        report = render(output, config["data"])
        if output.is_relative_to(config["outputs"]):
            config["outputs"].mkdir(parents=True, exist_ok=True)
            marker = config["outputs"] / "latest.tmp"
            marker.write_text(json.dumps({"run": str(output.relative_to(config["outputs"]))}))
            marker.replace(config["outputs"] / "latest.json")
        print("시각화:", report)
        if args.open:
            import webbrowser
            webbrowser.open(report.as_uri())
    elif args.command == "report":
        from evct.report import render
        output = args.result_dir.resolve() if args.result_dir else latest_result(config)
        report = render(output, config["data"])
        print("시각화:", report)
        if args.open:
            import webbrowser
            webbrowser.open(report.as_uri())

if __name__ == "__main__":
    # Training options belong to the shared trainer, not the evaluation parser.
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        configure()
        from evct.retrain import main as train
        sys.argv.pop(1)
        train()
    else:
        main()
