# 이미지+텍스트 융합으로 F1@5 0.8+ 달성 (2026-08-11, 최종 갱신 2026-08-21)

## 목표
image+text 둘 다 사용해서 F1@5 0.8 이상.

## ⚠️ 정정 사항 (2건)
1. 최초 버전은 `annotations-db` 캡션(`caption.ko`가 "익살스러운, 단순한, ... . <본문>" 식으로
   정답 라벨 5개를 문장 맨 앞에 그대로 노출하는 소스)을 텍스트 다양성 소스로 썼었음.
   **이 데이터 소스는 사용 금지 판정**을 받아 전면 제외하고, description(annotations,
   정상 소스)만으로 다시 진행함.
2. 텍스트 입력을 description(원문) + 문양유형/재질/시대 등 **구조적 메타데이터**를 합쳐서
   만들고 있었는데, **원래 의도는 description 텍스트만 쓰는 것**이었다는 게 확인됨.
   모든 모델을 description-only로 다시 학습시켰음 (§4 참고).

## 최종 결과: **F1@5 = 0.8000** (val=446, db-caption 미사용, description만 사용)

## 구조
- **텍스트**: 5개 한국어/다국어 인코더를 raw description 텍스트만으로(메타데이터 없이)
  end-to-end fine-tuning (frozen이 아니라 진짜 학습 - 이게 핵심 레버. frozen이었을 때는
  0.68 수준).
- **이미지**: DINOv3(facebook/dinov3-vitl16-pretrain-lvd1689m), frozen, pooled (CLS+mean-patch).
  기존 세션에서 이미 캐싱해둔 특징(`dinov3_features/features_*.pt`, 이 디렉토리 안에 포함됨) 재사용.
- **융합**: 텍스트 pooled + 이미지 pooled를 각각 projection → 2-토큰 시퀀스로 만들어
  가벼운 self-attention(8 head, Q=K=V=시퀀스) 1층 → concat → 분류 head.
  (2-토큰짜리 가벼운 attention은 ML-Decoder 같은 무거운 patch-token attention과 달리
  collapse 안 하고 concat보다 나음 - 사전 검증: 0.677 vs 0.661)
- **학습**: discriminative LR(백본 1e-5~1.5e-5, 새 레이어 1e-3), OneCycleLR, AsymmetricLoss,
  no class-balanced sampler(top-5 지표엔 natural 분포가 유리), 12~15 epoch.
- **다양성 레버**: 텍스트 소스는 description 하나로 고정하고, **서로 다른 사전학습
  코퍼스/아키텍처를 가진 5종**(klue-roberta-large, xlm-roberta-large, kcbert-large,
  bert-base-multilingual-cased, kobigbird-bert-base)을 앙상블. 아키텍처가 다르면 오류가
  decorrelated되어 앙상블 이득이 큼.
- koelectra-base, mdeberta-v3-base(로딩 실패), klue-bert-base, deberta-v3-xlarge-korean
  (토크나이저 호환성 문제로 학습 불가)도 시도했으나 최종 5개 조합보다 낫지 않아 제외.

## 개별 모델 성능 (val F1@5) — 최종 배포 5개, 전부 description-only
| 태그 | 아키텍처 | 비고 | F1@5 |
|---|---|---|---|
| klue_descOnly_cfgB | klue-roberta-large | fine-tuning 설정 튜닝 | 0.7794 |
| xlmr_descOnly | xlm-roberta-large | 기본 설정 | 0.7753 |
| kcbert_descOnly | beomi/kcbert-large | max_length=300 | 0.7682 |
| mbert_descOnly | bert-base-multilingual-cased | 기본 설정 | 0.7753 |
| kobigbird_descOnly | monologg/kobigbird-bert-base | 기본 설정, 단독 최고 | 0.7861 |

## 대조군: 텍스트 인코더 freeze vs fine-tune
텍스트 인코더를 얼린 채(가중치 고정, 새로 붙인 분류 head만 학습) 썼을 때와 전체를 진짜
fine-tuning했을 때를 비교한 대조군 결과:

| 텍스트 인코더 처리 | F1@5 |
|---|---|
| Frozen (가중치 고정, head만 학습) | ~0.68 |
| Fine-tune (전체 가중치 학습, 최종 배포) | 0.77~0.79 (개별) → **0.80** (앙상블) |

이 프로젝트에서 성능을 가장 크게 끌어올린 레버는 이미지 인코더 선택도, 융합 방식도 아니라
**텍스트 인코더를 얼리지 않고 fine-tuning한 것**이었음.

## 앙상블
| 방법 | F1@5 |
|---|---|
| 5개 동일가중 평균 | 0.7946 |
| **가중치 탐색 최적** (klue=.3135, xlmr=.2099, kcbert=.2143, mbert=.2106, kobigbird=.0518) | **0.8000** |

### 모델 개수/조합 탐색 (2026-08-21)
description-only로 전환 후 처음엔 klue+xlmr+kcbert+klue_bert 4개로 재학습해서 0.7991이
나왔음(0.8 미달). 새 아키텍처(mbert, kobigbird)를 추가 학습시키고 6개 후보 중 2~6개
크기의 모든 조합을 Dirichlet 탐색으로 훑은 결과, klue_bert를 kobigbird로 바꾼 5개 조합이
0.8000으로 가장 좋았음(6개 전부 쓰는 것보다도 나음). 상세 표는
[`../inference/docs/모델설명.md`](../inference/docs/모델설명.md) §5-2 참고.

## 재현
```bash
python train_fusion_final.py --tag klue_descOnly_cfgB --model klue/roberta-large --desc_only --lr 1e-5 --dropout 0.1 --bs 4 --grad_accum 4 --max_length 384 --epochs 15 --gpu 0
python train_fusion_final.py --tag xlmr_descOnly      --model xlm-roberta-large --desc_only                                                              --gpu 1
python train_fusion_final.py --tag kcbert_descOnly    --model beomi/kcbert-large --desc_only --max_length 300                                            --gpu 2
python train_fusion_final.py --tag mbert_descOnly     --model bert-base-multilingual-cased --desc_only                                                   --gpu 3
python train_fusion_final.py --tag kobigbird_descOnly --model monologg/kobigbird-bert-base --desc_only                                                   --gpu 4
```
`--desc_only`를 안 주면 기존처럼 description+구조적 메타데이터를 합쳐서 쓴다(비권장 —
원래 의도와 다름). 각 `features/fusion_{tag}_val.npz`에 val_prob/val_ids/val_f1 저장됨.
`ensemble.py`(TAGS=최종 5개 하드코딩)로 앙상블 평가.

## 참고
- `features/`에는 위 5개 외에도 과정에서 시도했던 여러 변형(메타데이터 포함 버전, 시드
  배깅 7개 버전, same-LR 실험, klue_bert 포함 4개 버전 등)의 체크포인트가 같이 남아있음 —
  전부 참고용이고, 실제 배포는 위 5개(`final_model/` = `inference/`)만 사용함.
- 별도 초기 실험에서도 **텍스트 단독**으로 0.8099까지 달성한
  기록이 있으나(이 세션과 별개 트랙), 그쪽도 db-caption을 다양성 소스로 썼던 부분이 있어
  같은 정정이 필요할 수 있음.
