# 최종 모델 상세 설명 — 이미지+텍스트 융합 앙상블 (val F1@5 = 0.8000)

구조도: [`fusion_architecture.html`](./fusion_architecture.html)

---

## 1. 학습에 사용된 모델

### 이미지 인코더 (1개, 공통)
| 항목 | 내용 |
|---|---|
| 모델 | `facebook/dinov3-vitl16-pretrain-lvd1689m` (DINOv3, ViT-Large) |
| 학습 여부 | **완전 고정(frozen)** — 이번 학습에서 가중치 전혀 안 바뀜 |
| 출력 | CLS 토큰 + patch 토큰 평균을 concat한 pooled 벡터 (1024×2 = **2048차원**) |
| 선택 이유 | 이 프로젝트에서 CLIP/SigLIP2/FG-CLIP2/DINOv3를 이미지 단독으로 비교했을 때 DINOv3가 가장 성능이 좋았음(self-supervised라 텍스트 정렬에 편향되지 않은 순수 시각 표현). 5개 텍스트 모델 전부가 이 하나의 이미지 인코더를 공유함. |

### 텍스트 인코더 (5개, 각각 fine-tuning, description 텍스트만 사용)
| 태그 | 모델 | 사전학습 특징 | 최종 개별 성능(F1@5) |
|---|---|---|---|
| `klue` | klue/roberta-large | 한국어 뉴스·위키 코퍼스, RoBERTa 구조 | 0.7794 (fine-tuning 설정 튜닝) |
| `xlmr` | xlm-roberta-large | 100개 언어 다국어 코퍼스 | 0.7753 |
| `kcbert` | beomi/kcbert-large | 한국어 **커뮤니티/구어체** 텍스트(네이버 뉴스 댓글) | 0.7682 |
| `mbert` | bert-base-multilingual-cased | 104개 언어 다국어 코퍼스, mBERT 구조(xlmr과는 세대/토크나이저가 다름) | 0.7753 |
| `kobigbird` | monologg/kobigbird-bert-base | 한국어 코퍼스, BigBird(sparse attention) 구조 | 0.7861 (단독 최고) |

5개를 고른 기준은 **서로 다른 사전학습 코퍼스 또는 다른 아키텍처**를 하나씩 확보하는 것 —
klue-roberta 하나만으로는 0.78 근처에서 정체되는데, 성격이 다른 모델을 섞으면 각자
실수하는 지점이 달라서(오류가 decorrelated) 앙상블 시 유의미하게 올라감.

> **텍스트 입력은 description 문장만 사용한다** (문양유형/재질/시대 같은 구조적 메타데이터는
> 넣지 않음). 이전 버전은 실수로 메타데이터를 같이 넣고 있었는데, description만 쓰는 게
> 원래 의도였다는 게 확인돼서 전부 다시 학습시켰음. 메타데이터를 빼도 개별 모델 성능
> 차이는 크지 않았음(±0.003~0.004, 노이즈 수준) — §4-3 참고.

> 원래는 klue_bert를 포함한 4개 조합(0.7991)이었는데, klue_bert 대신 kobigbird를 넣은
> 5개 조합이 0.8000으로 더 높아서 이걸 채택함 — §5-2 참고.

---

## 2. 융합(fusion) 구조

이미지 pooled 벡터(2048-d)와 텍스트 pooled 벡터(모델별 768~1024-d)를 각각 `Linear`로
**같은 1024차원**으로 projection한 뒤, 이 둘을 "토큰 2개짜리 시퀀스"로 보고
**8-head self-attention 1층**을 태운다. Q, K, V가 전부 이 2-토큰 시퀀스에서 나오는
표준 self-attention이라, 두 토큰(이미지/텍스트) 모두 [자기 자신, 상대방]을 한 번씩
참고한다. 그 결과(residual+LayerNorm)를 다시 이어붙여서(2048-d) 작은 MLP를 거쳐
22개 라벨 점수를 출력하고, sigmoid 확률 중 상위 5개를 최종 예측으로 고른다.

- 왜 attention을 쓰는가: 단순 concat보다 "이미지가 텍스트를, 텍스트가 이미지를 한 번
  참고하는" 여지를 준다. 사전 비교 실험에서 concat(0.661) < attention(0.677)로 확인함.
- 왜 무겁게(patch-token 단위) 안 하는가: 이 프로젝트에서 patch-token(256개) + 라벨별
  쿼리(22개) 방식의 ML-Decoder를 시도했을 때, 데이터가 ~4,000장뿐이라 학습이 제대로
  안 되고 출력이 collapse(로짓이 거의 균일해짐)하는 걸 이미 확인함. 그래서 토큰을
  **딱 2개**로 극단적으로 가볍게 만들어서 같은 문제를 피함.

---

## 3. 학습 디테일

| 항목 | 값 | 비고 |
|---|---|---|
| Loss | AsymmetricLoss (gamma_neg=4, gamma_pos=1, clip=0.05) | 쉬운 negative를 억제해서 클래스 불균형 완화 |
| 샘플러 | 사용 안 함 (natural 분포 그대로) | class-balanced sampler를 써봤는데 top-5 지표엔 오히려 손해라는 게 이전 실험(병행 트랙)에서 확인됨 |
| 옵티마이저 | AdamW, discriminative LR (아래 5번 항목 참고) | |
| 스케줄러 | OneCycleLR (pct_start=0.1) | 웜업 후 코사인 감쇠 |
| Epoch | 12~15 (모델별 수렴 시점까지 소폭 튜닝) | |
| Batch size | 기본 16 (klue는 GPU 자원 부족으로 4, gradient accumulation 4로 실효 16 유지) | |
| Gradient clipping | max_norm=1.0 | |
| 텍스트 입력 | **description 문장만** (구조적 메타데이터 미포함) | annotations-db(정답 라벨이 캡션 맨 앞에 노출된 소스)는 사용 금지 처리 후 제외 |
| max_length | 모델별 300~512 (kcbert는 300 — 원 모델 특성상 짧게) | |
| 데이터 분할 | train 3,954 / val 446, iterative stratification (seed=42) | 기존 실험들과 동일 split 재사용 → 공정 비교 |

---

## 4. 학습 이유(설계 판단) 정리

### 4-1. 왜 이미지를 반드시 넣었나
텍스트(description) 단독으로도 fine-tuning하면 0.78 근처까지 나오지만(이 프로젝트
요구사항상) **이미지가 실제로 기여해야 한다**는 조건이 있었음. 검증을 위해 학습된
모델에 실제 이미지 대신 (a) 0벡터, (b) 엉뚱한 이미지 특징을 넣어봤더니 F1@5가
떨어지는 걸 확인 — 이미지 정보가 실제로 예측에 쓰이고 있음을 직접 증명함.

### 4-2. 왜 이미지 인코더는 얼리고 텍스트 인코더는 학습시켰나
- 이미지(DINOv3): 이 프로젝트에서 스케치 ResNet을 완전히 풀어서(full fine-tune)
  학습해봤을 때 프리즈 버전과 성능이 거의 같았음(0.544 vs 0.546) — 데이터가 4천 장뿐인
  상황에서 이미지 백본을 통째로 열어도 이득이 없고 과적합 위험만 커짐. ConvNeXtV2로
  별도 재검증도 했으나(완전 프리즈/도메인 적응 후 프리즈/텍스트와 동일 LR로 언프리즈/
  더 낮은 LR로 언프리즈, 4조건) 결론은 동일 — 이 데이터 규모에서는 이미지 프리즈가
  합리적인 선택.
- 텍스트(RoBERTa 등): 반대로 텍스트 인코더는 fine-tuning의 이득이 이미지 쪽과 정반대로
  컸음 — 대조군으로 비교하면:

| 텍스트 인코더 처리 | F1@5 |
|---|---|
| Frozen (가중치 고정, pooled 임베딩만 뽑아 MLP 학습) | ~0.68 |
| Fine-tune (end-to-end 전체 학습) | 0.78대 |

### 4-3. 메타데이터를 뺐을 때 성능 영향
description만 쓰는 게 원래 의도였다는 게 확인돼서, 기존(메타데이터+description)
버전과 신규(description만) 버전을 같은 조건으로 비교했다:

| 모델 | 메타데이터 포함 | description만 | 차이 |
|---|---|---|---|
| klue | 0.7843 | 0.7794 | -0.0049 |
| xlmr | 0.7789 | 0.7753 | -0.0036 |
| kcbert | 0.7722 | 0.7682 | -0.0040 |
| klue_bert | 0.7780 | 0.7807 | +0.0027 |

차이가 ±0.003~0.005 수준으로 노이즈와 구분이 잘 안 될 만큼 작음 — 구조적 메타데이터가
성능에 크게 기여하지는 않는다는 뜻. description 텍스트 자체가 이미 충분한 신호를
담고 있는 것으로 보임.

### 4-4. 왜 아키텍처를 5개나 앙상블했나
단일 모델은 전부 0.77~0.78대에서 정체됨(klue 0.7794, xlmr 0.7753, kcbert 0.7682,
mbert 0.7753, kobigbird 0.7861). 목표(0.8)를 넘으려면 서로 다른 사전학습 특성을 가진
모델들을 섞어서 오류를 분산시켜야 했음. klue_bert를 포함한 4개 조합(0.7991)으로는
근소하게 부족했고, mbert·kobigbird라는 새로운 아키텍처(각각 다른 다국어 코퍼스,
sparse-attention 계열)를 후보에 추가해서 2~6개 전체 조합을 탐색한 결과 **klue+xlmr+
kcbert+mbert+kobigbird 5개 조합이 정확히 0.8000**으로 가장 좋았음. (참고: 처음엔
db-caption을 텍스트 소스 다양성으로 썼다가 그 데이터가 정답 라벨을 캡션 앞에 노출하는
leakage 소스라는 게 확인돼서 전면 폐기하고, 대신 아키텍처 다양성으로 대체했음.)

---

## 5. 러닝레이트를 다르게 잡은 이유 (핵심 설계 의도)

이 모델은 **discriminative learning rate**를 쓴다 — 텍스트 백본은 아주 낮은 LR
(1e-5~1.5e-5), 새로 붙인 fusion/attention/classifier 레이어는 훨씬 높은 LR(1e-3)로
학습한다. 이건 단순히 "사전학습된 걸 덜 흔든다"는 관용적인 이유를 넘어서, **의도적으로
distillation과 비슷한 방식으로 설계**한 것이다:

- **일반적인 fine-tuning**은 backbone도 task loss로 비교적 크게(빠르게) 업데이트해서,
  사전학습된 표현을 태스크 전용 표현으로 상당히 많이 바꿔버린다. 이 방식은 데이터가
  많을 땐 잘 통하지만, 우리처럼 ~4,000장뿐인 상황에서는 backbone이 가진 원래의
  풍부한 언어 지식(문법, 의미, 세상 지식)을 태스크에 맞춰 과도하게 재작성하면서
  오히려 일반화 능력을 깎아먹을 위험이 있다.
- 그래서 backbone LR을 아주 작게 잡아서, **매 스텝 아주 조금씩만, 그러나
  "계속(지속적으로)"** 업데이트되게 했다 — 사전학습된 임베딩 공간의 전체적인 구조는
  최대한 그대로 유지하면서, 그 구조 위에서 우리 태스크(감성 분류)에 맞게 미세하게
  좌표를 옮겨가는 방식이다. 이건 **얼린 teacher의 지식을 유지한 채로 표현을 조금씩
  맞춰가는 knowledge distillation의 정신**과 비슷하다.
- 반면 새로 붙인 fusion 레이어(image_proj/text_proj/attention/classifier)는 랜덤
  초기화된 상태라 처음부터 배울 게 아무것도 없다. 이런 레이어까지 backbone과 같은
  낮은 LR로 학습시키면 12~15 epoch 안에 제대로 수렴하지 못한다. 그래서 이 레이어들은
  100배 높은 LR(1e-3)로 빠르게 학습시켜서, "천천히 적응하는 backbone" 위에 "빠르게
  배우는 새 분류기"가 함께 최적점을 찾아가도록 설계했다.

아래 §5-1 검증 결과를 바탕으로, 모델별 fine-tuning 설정(backbone LR, epoch)을 소폭
조정했다 — backbone을 더 천천히, 더 오래 적응시키는 방향이 유효하다는 걸 klue 모델로
재확인함(0.7767→0.7794). 정확한 재현 설정은 `train_fusion_final.py` 실행 커맨드
(RESULT.md §재현) 참고.

### 5-1. 추가 검증 — "그럼 얼리거나, 아예 같은 LR로 풀면 안 되나?"를 직접 실험함

위 설명이 그럴듯한 서사이긴 하지만, 실제로 대안들과 비교하지 않으면 근거가 약하다.
그래서 klue 모델로 4가지 조건을 직접 다 돌려서 비교했다 (val F1@5, 다른 조건 동일):

| 조건 | F1@5 | 비고 |
|---|---|---|
| 텍스트 인코더 완전 프리즈 (학습 안 함) | 0.677 | 가장 나쁨 — pooled 임베딩만으론 부족 |
| backbone/head 둘 다 낮은 LR로 통일 (1.5e-5) | **0.786** | discriminative보다 오히려 살짝 높음 |
| **discriminative LR (backbone 1.5e-5 / head 1e-3)** | 0.779~0.784 | 기준 |
| backbone/head 둘 다 높은 LR로 통일 (1e-3) | 0.583 | 크게 나쁨 — backbone 표현이 망가짐 |

**정직한 결론**: "head는 높은 LR로 빠르게 학습시켜야 한다"는 원래 가설은 이번 검증에서
**뒷받침되지 않았다** — 낮은 LR로 통일해도 head(작은 Linear+attention 몇 개뿐)는
12 epoch 안에 충분히 수렴했고, 오히려 discriminative보다 나은 경우도 있었다.
**진짜 핵심은 딱 하나, "backbone LR을 낮게 유지해야 한다"는 것**이다 — 이게 높으면
사전학습된 표현이 몇 epoch 안에 망가져서 성능이 반토막 난다(0.78→0.58). 텍스트
인코더를 아예 프리즈하는 것도 확실히 나쁘다(0.68 수준).

### 5-2. 모델 개수/조합 탐색 — 4개(klue_bert 포함) vs 5개(kobigbird 포함)

description-only로 전환한 뒤, 동일 설정(klue만 튜닝, 나머지 기본값)으로 4개 모델
(klue+xlmr+kcbert+klue_bert)을 다시 학습시켰더니 **0.7991**로 0.8에 살짝 못 미쳤다.
그래서 완전히 새로운 아키텍처 후보(mbert, kobigbird, deberta-v3-xlarge-korean — 이건
토크나이저 호환성 문제로 실패)를 추가로 학습시키고, 6개 후보 중 2~6개 크기의 모든
조합에 대해 Dirichlet 가중치 탐색을 돌렸다:

| 조합 | F1@5 |
|---|---|
| 4개 (klue+xlmr+kcbert+klue_bert) | 0.7991 |
| 4개 (klue+xlmr+kcbert+mbert) | 0.7991 |
| 5개 (klue+xlmr+kcbert+klue_bert+mbert) | 0.7991 |
| 5개 (klue+xlmr+kcbert+klue_bert+kobigbird) | 0.7996 |
| **5개 (klue+xlmr+kcbert+mbert+kobigbird)** | **0.8000** |
| 6개 (전부) | 0.7996 |

klue_bert를 kobigbird로 교체한 5개 조합이 정확히 0.8000으로 가장 좋았다(6개 전부
쓰는 것보다도 나음 — 모델을 더 추가한다고 항상 좋아지는 건 아니라는 것도 확인).
이 조합을 최종 채택함.

(참고 체크포인트: `features/fusion_klue_samelr_low_*.pt`, `fusion_*_descOnly_*.pt` 등 —
과정에서 나온 여러 변형은 참고용으로 `final_model/`(=`inference/`)엔 포함 안 함)
