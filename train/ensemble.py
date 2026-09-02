"""최종 배포 5개 fusion 모델(아키텍처 5종, description 텍스트만 사용, db-caption 미사용)의
val 확률을 앙상블해서 F1@5 평가. 가중치는 Dirichlet 랜덤서치로 탐색(val 446장 기준).
final_model/config.json에 저장된 가중치와 동일한 조합을 재현한다."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset as D  # noqa: E402

TAGS = ["klue_descOnly_cfgB", "xlmr_descOnly", "kcbert_descOnly", "mbert_descOnly", "kobigbird_descOnly"]
N_SEARCH = 50000
SEARCH_SEED = 2


def main():
    recs = D.load_records()
    vocab = D.build_vocab(recs)
    _, va = D.split_records(recs)
    va_by_id = {r["id"]: r for r in va}

    models = {tag: np.load(f"features/fusion_{tag}_val.npz", allow_pickle=True) for tag in TAGS}
    ids = models[TAGS[0]]["val_ids"]
    for tag in TAGS:
        assert list(models[tag]["val_ids"]) == list(ids), f"{tag} id 불일치"
    va_ordered = [va_by_id[i] for i in ids]

    for tag, d in models.items():
        print(f"{tag}: F1={float(d['val_f1']):.4f}")

    eq = sum(models[t]["val_prob"] for t in TAGS) / len(TAGS)
    print("동일가중 평균:", D.top5_f1(eq, va_ordered, vocab))

    rng = np.random.default_rng(SEARCH_SEED)
    best = (-1, None)
    for _ in range(N_SEARCH):
        w = rng.dirichlet(np.ones(len(TAGS)))
        ens = sum(wi * models[t]["val_prob"] for wi, t in zip(w, TAGS))
        f1 = D.top5_f1(ens, va_ordered, vocab)
        if f1 > best[0]:
            best = (f1, dict(zip(TAGS, w.round(4))))
    print(f"최적 가중치: {best[1]}  F1={best[0]:.4f}")


if __name__ == "__main__":
    main()
