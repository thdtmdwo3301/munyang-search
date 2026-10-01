"""Run real DINOv3 + text-fusion inference for four cases and write a static report."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import random
import secrets
import time
import webbrowser
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
STRATA = ["5/5", "4/5", "3/5", "2/5 이하"]


def choose_cases(cases: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    selected = []
    for stratum in STRATA:
        pool = sorted((case for case in cases if case["stratum"] == stratum), key=lambda case: case["id"])
        if not pool:
            raise RuntimeError(f"사례 구간이 비어 있습니다: {stratum}")
        selected.append(rng.choice(pool))
    return selected


def latest_selection(outputs_dir: Path) -> list[str] | None:
    candidates = sorted(outputs_dir.glob("*/results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return [case["id"] for case in payload["cases"]]
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    return None


def auto_selection(cases: list[dict], outputs_dir: Path) -> tuple[int, list[dict]]:
    previous = latest_selection(outputs_dir)
    for _ in range(100):
        seed = secrets.randbits(63)
        selected = choose_cases(cases, seed)
        if [case["id"] for case in selected] != previous:
            return seed, selected
    raise RuntimeError("새 사례 조합을 선택하지 못했습니다")


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def pills(values: list[str], css_class: str) -> str:
    return "".join(f'<span class="pill {css_class}">{html.escape(value)}</span>' for value in values)


def render_report(payload: dict, manifest_dir: Path) -> str:
    cards = []
    for index, case in enumerate(payload["cases"], start=1):
        image_uri = image_data_uri(manifest_dir / case["image"])
        cards.append(f"""
        <article class="card">
          <header><div><b>{index:02d} · 앙상블 정답 {case['hit_count']}/5</b><small>{html.escape(case['id'])}</small></div></header>
          <div class="card-body">
            <section class="source">
              <img src="{image_uri}" alt="{html.escape(case['id'])} 문양 이미지">
              <h3>원설명문</h3>
              <p>{html.escape(case['description'])}</p>
            </section>
            <section class="result">
              <dl>
                <dt>맞힌 정답</dt><dd>{pills(case['correct'], 'ok') or '<span class="empty">없음</span>'}</dd>
                <dt>놓친 정답</dt><dd>{pills(case['missed'], 'miss') or '<span class="empty">없음</span>'}</dd>
                <dt>오답</dt><dd>{pills(case['errors'], 'error') or '<span class="empty">없음</span>'}</dd>
              </dl>
              <div class="divider"></div>
              <h3>정답 5개</h3><p>{' · '.join(map(html.escape, case['gt']))}</p>
              <h3>앙상블 Top-5</h3>
              <ol>{''.join(f'<li><span>{html.escape(item["label"])}</span><strong>{item["score"]:.3f}</strong></li>' for item in case['top5'])}</ol>
            </section>
          </div>
        </article>""")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>전통문양 앙상블 정성 평가</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7f9;color:#173849;font-family:Pretendard,"Malgun Gothic","Apple SD Gothic Neo",sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 8px;font-size:32px}}.lead{{margin:0 0 24px;color:#627985}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}}.card{{background:#fff;border:1px solid #cbd8de;border-top:7px solid #2f6f8f;border-radius:12px;overflow:hidden}}
.card header{{padding:18px 20px 10px;font-size:21px}}small{{display:block;margin-top:8px;color:#718792;font-size:13px;font-weight:400}}
.card-body{{display:grid;grid-template-columns:330px 1fr;gap:24px;padding:8px 20px 22px}}.source img{{width:100%;height:255px;object-fit:contain;background:#fff}}
h3{{font-size:15px;margin:15px 0 7px}}p{{font-size:14px;line-height:1.65;margin:0}}dl{{margin:0}}dt{{color:#718792;font-size:13px;margin:10px 0 6px}}dd{{margin:0;display:flex;flex-wrap:wrap;gap:6px}}
.pill{{display:inline-block;padding:5px 8px;border-radius:7px;font-weight:700;font-size:13px}}.ok{{background:#e1f4ea;color:#157347}}.miss{{background:#fde9e7;color:#b93f38}}.error{{background:#fff0df;color:#b55a19}}.empty{{color:#718792}}
.divider{{height:1px;background:#dce6ea;margin:16px 0}}ol{{list-style:none;padding:0;margin:0}}li{{display:flex;justify-content:space-between;border-bottom:1px solid #edf1f3;padding:7px 0;font-size:14px}}li strong{{color:#007681}}
footer{{color:#718792;font-size:13px;margin-top:18px}}@media(max-width:980px){{.grid{{grid-template-columns:1fr}}}}@media(max-width:650px){{main{{padding:14px}}.card-body{{grid-template-columns:1fr}}}}
</style></head><body><main>
<h1>기존 검증 F1@5 80.00% · 5종 앙상블 정성 평가</h1>
<p class="lead">DINOv3 이미지 특징을 새로 추출하고 원설명문과 융합한 실제 추론 결과 · 실행 사례 4개</p>
<section class="grid">{''.join(cards)}</section>
<footer>device {html.escape(payload['device'])} · seed {payload['seed']} · 총 추론 {payload['elapsed_seconds']:.1f}초</footer>
</main></body></html>"""


def main() -> None:
    from inference.infer import EnsembleModel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=ROOT / "inference/weights")
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--open", action="store_true", dest="open_report")
    args = parser.parse_args()

    manifest_path = args.cases.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs_dir = ROOT / "outputs"
    seed_value = args.seed
    if seed_value is None and os.environ.get("EVALUATION_SEED"):
        seed_value = int(os.environ["EVALUATION_SEED"])
    if seed_value is None:
        seed_value, selected = auto_selection(manifest["cases"], outputs_dir)
    else:
        selected = choose_cases(manifest["cases"], seed_value)

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = outputs_dir / f"static_eval_{stamp}_{secrets.token_hex(3)}"
    output_dir.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    model = EnsembleModel(
        device=args.device,
        weights_dir=args.weights_dir,
        local_files_only=args.local_files_only,
    )
    result_cases = []
    for case in selected:
        labels, scores = model.predict(str(manifest_path.parent / case["image"]), case["description"])
        gt_set = set(case["gt"])
        prediction_set = set(labels)
        result_cases.append({
            **case,
            "top5": [{"label": label, "score": float(scores[label])} for label in labels],
            "hit_count": len(gt_set & prediction_set),
            "correct": [label for label in labels if label in gt_set],
            "missed": [label for label in case["gt"] if label not in prediction_set],
            "errors": [label for label in labels if label not in gt_set],
        })
    payload = {
        "status": "PASS",
        "mode": "real DINOv3 feature extraction and five-model ensemble inference",
        "device": model.device,
        "seed": seed_value,
        "elapsed_seconds": time.perf_counter() - started,
        "cases": result_cases,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "report.html"
    report_path.write_text(render_report(payload, manifest_path.parent), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "report": str(report_path), "seed": seed_value}, ensure_ascii=False))
    if args.open_report:
        webbrowser.open(report_path.as_uri())


if __name__ == "__main__":
    main()
