"""Standalone HTML report from actual evaluation outputs and source annotations."""
import base64
import html
import json
from pathlib import Path

def render(result_dir, data_root):
    result_dir, data_root = Path(result_dir), Path(data_root)
    summary = json.loads((result_dir / "results.json").read_text())
    sources = {}
    for path in sorted((data_root / "annotations").glob("*.json")):
        item = json.loads(path.read_text())
        content = item.get("contents", {})
        sources[str(item.get("identifier", path.stem))] = {
            "image": data_root / "images" / content.get("image", {}).get("file_name", ""),
            "gt": content.get("emotion", []), "text": item.get("description", "")}
    h = html.escape
    cards, metrics = [], []
    for mode, title, color in (("image", "이미지 단독", "#2463eb"),
                               ("multimodal", "이미지 + 원설명문", "#009e73")):
        file = result_dir / (mode + "_predictions.jsonl")
        if not file.exists():
            continue
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        if len(rows) != summary["validation_records"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Prediction count mismatch")
        hits = 0
        for row in rows:
            source = sources[row["id"]]
            if len(source["gt"]) != 5 or len(set(row["predicted_top5"])) != 5:
                raise ValueError("Invalid ground truth or prediction")
            correct = len(set(source["gt"]) & set(row["predicted_top5"]))
            if correct != row["correct_count"]:
                raise ValueError("Source data does not match saved predictions")
            hits += correct
        score = summary[mode + "_f1_at_5"] * 100
        if hits != summary[mode + "_correct_labels"] or abs(hits/(len(rows)*5)*100-score) > .0001:
            raise ValueError("Metric does not match predictions")
        metrics.append(f'<div class="metric"><strong>{h(title)}</strong><b>{score:.2f}%</b>'
            f'<div class="track"><div style="width:{score:.4f}%;background:{color}"></div></div>'
            f'<small>정답 라벨 {hits:,} / {len(rows)*5:,}</small></div>')
        # One deterministic example per nonempty correctness band.
        chosen = []
        for band in (5, 4, 3, 2):
            candidates = [r for r in rows if r["correct_count"] == band] if band > 2 else [r for r in rows if r["correct_count"] <= 2]
            if candidates:
                chosen.append(sorted(candidates, key=lambda r:r["id"])[0])
        for row in chosen:
            source = sources[row["id"]]
            mime = "image/png" if source["image"].suffix.lower() == ".png" else "image/jpeg"
            uri = "data:" + mime + ";base64," + base64.b64encode(source["image"].read_bytes()).decode()
            predicted = "".join(f'<li class="{"ok" if label in source["gt"] else "bad"}">'
                f'<span>{h(label)}</span><b>{score:.4f}</b></li>'
                for label, score in zip(row["predicted_top5"], row["scores"]))
            description = f'<h4>입력 설명문</h4><p>{h(source["text"])}</p>' if mode == "multimodal" else ""
            cards.append(f'<article class="card" data-mode="{mode}"><header>{h(title)} · 정답 {row["correct_count"]}/5'
                f'<small>{h(row["id"])}</small></header><div class="body"><img src="{uri}" alt="문양">'
                f'<section><h4>정답 감성어</h4><p>{h(" · ".join(source["gt"]))}</p>'
                f'<h4>예측 상위 5개</h4><ol>{predicted}</ol>{description}'
                '</section></div></article>')
    if not metrics:
        raise ValueError("No prediction results")
    condition = f'검증 {summary["validation_records"]}건'
    page = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>문양 감성분류 평가 결과</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:#123b53;font-family:"Noto Sans KR","Malgun Gothic",sans-serif}
main{max-width:1400px;margin:auto;padding:28px}h1{margin:0 0 12px}.muted,small{color:#607487}small{display:block;margin-top:8px}
.metrics,.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px;margin:24px 0}
.metric,.card{background:white;border:1px solid #d5e1eb;border-radius:10px;overflow:hidden}.metric{padding:24px}.metric b{float:right;font-size:24px}
.track{height:8px;background:#e5ebf0;margin:24px 0 16px}.track div{height:100%}
header{padding:18px;background:#073c59;color:white;font-weight:bold}header small{color:#d6e5ed}
.body{padding:18px;display:grid;grid-template-columns:180px 1fr;gap:18px}img{width:100%;height:220px;object-fit:contain}
h4{margin:12px 0 8px}p{font-size:14px;line-height:1.65}ol{padding:0;list-style:none}li{display:flex;justify-content:space-between;padding:6px;border-bottom:1px solid #e7edf3}
.ok{color:#008854}.bad{color:#ce382d}button{padding:9px 16px;background:white;border:1px solid #93afc2;border-radius:6px;cursor:pointer}
@media(max-width:850px){.metrics,.grid{grid-template-columns:1fr}.body{grid-template-columns:120px 1fr}}
</style></head><body><main><h1>문양 감성분류 평가 결과</h1>"""
    page += f'<p class="muted">{h(condition)}</p><div class="metrics">{"".join(metrics)}</div>'
    page += '<h2>정성 사례</h2><p class="muted">정답 개수 구간별 대표 사례 · 초록: 정답 / 빨강: 오답</p>'
    page += '<div><button onclick="filterCards(\'all\')">전체</button> <button onclick="filterCards(\'image\')">이미지 단독</button> <button onclick="filterCards(\'multimodal\')">이미지 + 설명문</button></div>'
    page += f'<div class="grid">{"".join(cards)}</div>'
    page += '<script>function filterCards(mode){document.querySelectorAll(".card").forEach(c=>c.hidden=mode!=="all"&&c.dataset.mode!==mode)}</script>'
    page += '</main></body></html>'
    target = result_dir / "report.html"
    target.write_text(page, encoding="utf-8")
    return target.resolve()
