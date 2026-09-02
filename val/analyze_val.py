"""최종 5개 앙상블 모델의 val(446장) 예측을 라벨별/샘플별로 뜯어보는 분석 스크립트.
../train/features/*_val.npz 를 읽는다 (../train/ensemble.py 로 학습 재현 후 생성됨).

사용법:
  python analyze_val.py
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
import dataset as D  # noqa: E402

FEATURES_DIR = HERE.parent / "train" / "features"
TAGS = ["klue_descOnly_cfgB", "xlmr_descOnly", "kcbert_descOnly", "mbert_descOnly", "kobigbird_descOnly"]
WEIGHTS = {"klue_descOnly_cfgB": 0.3135, "xlmr_descOnly": 0.2099, "kcbert_descOnly": 0.2143,
           "mbert_descOnly": 0.2106, "kobigbird_descOnly": 0.0518}

recs = D.load_records()
vocab = D.build_vocab(recs)
l2i = {l: i for i, l in enumerate(vocab)}
_, va = D.split_records(recs)
va_by_id = {r["id"]: r for r in va}

models = {tag: np.load(FEATURES_DIR / f"fusion_{tag}_val.npz", allow_pickle=True) for tag in TAGS}
ids = models[TAGS[0]]["val_ids"]
va_ordered = [va_by_id[i] for i in ids]

ens_prob = sum(WEIGHTS[t] * models[t]["val_prob"] for t in TAGS)
Y = D.multihot(va_ordered, vocab)

top5f1 = D.top5_f1(ens_prob, va_ordered, vocab)
print(f"전체 val F1@5 = {top5f1:.4f}  (n={len(va_ordered)})\n")

# ── 라벨별 성능 ──────────────────────────────────────────────
idx_top5 = np.argsort(-ens_prob, axis=1)[:, :5]
pred_mask = np.zeros_like(Y)
for i, row in enumerate(idx_top5):
    pred_mask[i, row] = 1

tp = ((pred_mask == 1) & (Y == 1)).sum(0)
fp = ((pred_mask == 1) & (Y == 0)).sum(0)
fn = ((pred_mask == 0) & (Y == 1)).sum(0)
support = Y.sum(0)

label_stats = []
for i, label in enumerate(vocab):
    prec = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
    rec = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    label_stats.append((label, int(support[i]), int(tp[i]), int(fp[i]), int(fn[i]), prec, rec, f1))

label_stats.sort(key=lambda x: x[-1])

print("=== 라벨별 성능 (F1 낮은 순) ===")
print(f"{'라벨':10s} {'GT수':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
for label, sup, t, f, n, p, r, f1 in label_stats:
    print(f"{label:10s} {sup:5d} {t:4d} {f:4d} {n:4d} {p:6.3f} {r:6.3f} {f1:6.3f}")

print(f"\n최저 F1 5개: {[x[0] for x in label_stats[:5]]}")
print(f"최고 F1 5개: {[x[0] for x in label_stats[-5:]]}")

# ── 샘플별 완전정답/부분정답/완전오답 분포 ──────────────────────────
per_sample_hits = (pred_mask * Y).sum(1)  # 0~5
dist = Counter(int(h) for h in per_sample_hits)
print("\n=== 샘플당 top5-hit 개수 분포 (5개 중 몇 개 맞았나) ===")
for k in sorted(dist):
    print(f"  {k}/5 맞음: {dist[k]}장 ({dist[k]/len(va_ordered)*100:.1f}%)")

# ── 모델간 예측 상관/다양성 확인 (top5 예측 집합의 자카드 유사도) ─────
print("\n=== 모델쌍별 top5 예측 자카드 유사도(평균) ===")
top5_sets = {}
for t in TAGS:
    idx = np.argsort(-models[t]["val_prob"], axis=1)[:, :5]
    top5_sets[t] = [set(row.tolist()) for row in idx]

import itertools
for a, b in itertools.combinations(TAGS, 2):
    jac = np.mean([len(top5_sets[a][i] & top5_sets[b][i]) / len(top5_sets[a][i] | top5_sets[b][i])
                    for i in range(len(va_ordered))])
    print(f"  {a:22s} vs {b:22s}: {jac:.3f}")

# ── 완전 오답(0/5) 샘플 몇 개 살펴보기 ─────────────────────────────
zero_hit_idx = [i for i, h in enumerate(per_sample_hits) if h == 0]
print(f"\n완전오답(0/5) 샘플 수: {len(zero_hit_idx)}")
for i in zero_hit_idx[:5]:
    r = va_ordered[i]
    gt = sorted(r["emotions"])
    pred = sorted([vocab[j] for j in idx_top5[i]])
    print(f"  id={r['id']}  GT={gt}")
    print(f"           예측={pred}")
