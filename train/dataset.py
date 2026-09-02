"""공통 데이터 유틸 (2026-07-24 emotion F1 실험).

- ETRI orig_4.4k_260519 annotations 에서 emotion(정답, 이미지당 정확히 5개) +
  구조적 메타데이터(patern_type/usage/material/temporal/technique/motif/form/meaning) +
  description(한국어 서술) 을 모두 읽는다.
- 평가는 top-5 set overlap (모든 샘플이 5개 라벨 -> P=R=F1).
- split 은 기존 실험(train_clip_emotion.split_records, iterative stratification, seed=42,
  val_ratio=0.1) 을 그대로 재사용해서 이전 결과들과 직접 비교 가능하게 한다.
- LEAKAGE 방지: description/메타데이터 문자열에서 22개 emotion 단어를 제거(redact)한 텍스트를
  기본으로 쓴다. (emotion 라벨을 텍스트에서 직접 읽어 맞히는 부정을 막기 위함)
"""
import json
import glob
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np

# Set MUNYANG_DATA_ROOT to the local dataset directory before training.
BASE = Path(os.environ.get("MUNYANG_DATA_ROOT", "data/orig_4.4k_260519"))
ANN = BASE / "annotations"
IMG = BASE / "images"
ANNDB = BASE / "annotations-db"


def load_db_captions():
    """annotations-db 의 풍부한 paraphrase 캡션(ko)을 source_id(M-id) 기준으로 로드.

    캡션은 'emo1, emo2, ... . <상세 서술>' 형태로 emotion 5개가 접두부에 붙어 있으니
    첫 문장(접두부)을 제거해 라벨 leakage 를 막는다."""
    caps = {}
    for p in sorted(glob.glob(str(ANNDB / "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        sid = d.get("info", {}).get("source_id")
        ko = d.get("annotation", {}).get("caption", {}).get("ko", "") or ""
        if not sid or not ko:
            continue
        # 첫 '. ' 이후만 사용(감성 접두부 제거)
        idx = ko.find(". ")
        caps[sid] = ko[idx + 2:] if 0 <= idx < 120 else ko
    return caps

SEED = 42
VAL_RATIO = 0.1


def load_records():
    recs = []
    for p in sorted(glob.glob(str(ANN / "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        c = d.get("contents", {})
        e = c.get("emotion", []) or []
        fn = c.get("image", {}).get("file_name")
        if len(e) != 5 or not fn:
            continue
        ip = IMG / fn
        if not ip.exists():
            continue
        recs.append({
            "id": d.get("identifier", Path(p).stem),
            "image_path": str(ip),
            "emotions": e,
            "description": d.get("description", "") or "",
            "ptype": c.get("patern_type", "") or "",
            "usage": c.get("patern_usage", "") or "",
            "material": c.get("material", "") or "",
            "temporal": c.get("temporal", "") or "",
            "technique": c.get("technique", "") or "",
            "artifact": c.get("artifact_name", "") or "",
            "motif": list(c.get("motif", []) or []),
            "form": list(c.get("form", []) or []),
            "meaning": list(c.get("meaning", []) or []),
        })
    return recs


def build_vocab(records):
    return sorted({e for r in records for e in r["emotions"]})


def split_records(records, val_ratio=VAL_RATIO, seed=SEED):
    """train_clip_emotion.split_records 와 동일한 iterative stratification.

    기존 실험과 정확히 같은 train/val 를 얻기 위해 로직을 복제한다."""
    vocab = build_vocab(records)
    l2i = {l: i for i, l in enumerate(vocab)}
    sample_labels = [{l2i[e] for e in r["emotions"]} for r in records]
    n = len(records)
    n_val = max(1, round(n * val_ratio))
    cap = {"train": n - n_val, "val": n_val}
    total = Counter(j for s in sample_labels for j in s)
    desired = {"train": {j: c * (1 - val_ratio) for j, c in total.items()},
               "val": {j: c * val_ratio for j, c in total.items()}}
    rng = random.Random(seed)
    remaining = set(range(n))
    assign = {}
    while remaining:
        lc = Counter()
        for i in remaining:
            for j in sample_labels[i]:
                lc[j] += 1
        mn = min(lc.values())
        cands = [j for j, c in lc.items() if c == mn]
        j = rng.choice(cands)
        ex = [i for i in remaining if j in sample_labels[i]]
        rng.shuffle(ex)
        for i in ex:
            f = max(cap, key=lambda f: (desired[f].get(j, 0.0), cap[f]))
            assign[i] = f
            cap[f] -= 1
            for jj in sample_labels[i]:
                desired[f][jj] = desired[f].get(jj, 0.0) - 1
            remaining.discard(i)
    train = [records[i] for i in range(n) if assign[i] == "train"]
    val = [records[i] for i in range(n) if assign[i] == "val"]
    return train, val


def multihot(records, vocab):
    l2i = {l: i for i, l in enumerate(vocab)}
    Y = np.zeros((len(records), len(vocab)), dtype=np.float32)
    for i, r in enumerate(records):
        for e in r["emotions"]:
            Y[i, l2i[e]] = 1.0
    return Y


def top5_f1(prob, records, vocab):
    """prob: [N,22] 점수 -> 상위 5개 예측. 모든 샘플 정답 5개라 P=R=F1."""
    idx = np.argsort(-prob, axis=1)[:, :5]
    l2i = {l: i for i, l in enumerate(vocab)}
    tot = 0.0
    for i, r in enumerate(records):
        gt = {l2i[e] for e in r["emotions"]}
        tot += len(gt & set(idx[i].tolist())) / 5.0
    return tot / len(records)


def redact(text, vocab):
    """텍스트에서 22개 emotion 단어를 제거해 라벨 leakage 차단."""
    for e in vocab:
        text = text.replace(e, "")
    return text


def build_doc(r, vocab=None, redact_emotions=True, desc_override=None):
    """메타데이터 + description 을 하나의 한국어 문서로 직렬화.

    emotion 라벨 자체는 절대 넣지 않는다. redact_emotions=True 면 description/메타데이터
    문자열에 우연히 들어간 emotion 단어도 제거한다. desc_override 가 주어지면 원본
    description 대신 그 텍스트를 '설명'에 쓴다(예: annotations-db 풍부한 캡션)."""
    desc = desc_override if desc_override is not None else r["description"]
    parts = [
        f"문양유형: {r['ptype']}",
        f"용도: {r['usage']}",
        f"재질: {r['material']}",
        f"시대: {r['temporal']}",
        f"기법: {r['technique']}",
        f"유물명: {r['artifact']}",
        f"모티프: {', '.join(r['motif'])}",
        f"형태: {', '.join(r['form'])}",
        f"의미: {', '.join(r['meaning'])}",
        f"설명: {desc}",
    ]
    doc = ". ".join(parts)
    if redact_emotions and vocab is not None:
        doc = redact(doc, vocab)
    return doc


if __name__ == "__main__":
    recs = load_records()
    vocab = build_vocab(recs)
    tr, va = split_records(recs)
    print(f"records={len(recs)} vocab={len(vocab)} train={len(tr)} val={len(va)}")
    print("sample doc:\n", build_doc(tr[0], vocab)[:400])
