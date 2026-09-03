# 최종 모델 - 인퍼런스 패키지

이미지(DINOv3) + 텍스트(5개 언어모델) 융합 앙상블. **val F1@5 = 0.8000**

공개 저장소에는 데이터·DINO 특징·체크포인트를 넣지 않았으므로, 자산을 별도로 준비해야 실제 실행할 수 있습니다.

## 빠른 시작

저장소 루트에서 실행합니다.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r inference/requirements.txt
python inference/infer.py --image /path/to/image.jpg --description "이미지에 대한 한국어 설명"
```

Windows PowerShell에서는 가상환경을 `py -3.11 -m venv .venv`로 만든 뒤
`.\.venv\Scripts\Activate.ps1`로 활성화합니다.

`--description`에는 문양유형/재질/시대 같은 메타데이터 없이 **순수 설명 문장만** 넣으면 됨
(학습도 description 텍스트만으로 했음).

출력 예:
```
Top-5 예측:
  고전적인: 0.9301
  조화로운: 0.9059
  소박한: 0.8168
  단순한: 0.7357
  우아한: 0.7333
```

코드에서 직접 쓰려면:
```python
from infer import EnsembleModel
model = EnsembleModel()  # 최초 로드 시 5개 언어모델 + DINOv3 다 불러옴 (몇 초 소요)
labels, scores = model.predict(image_path, description, topk=5)
```

## 구성
```
inference/
├── requirements.txt   # 저장소 루트에서 pip install -r inference/requirements.txt
├── config.json        # 모델 구성/하이퍼파라미터/앙상블 가중치
├── infer.py            # 단일 이미지 추론 진입점
├── README.md           # 이 파일
├── weights/
│   ├── klue.pt         # klue-roberta-large (fine-tuning 설정 튜닝, val F1=0.7794)
│   ├── xlmr.pt         # xlm-roberta-large (val F1=0.7753)
│   ├── kcbert.pt       # kcbert-large (val F1=0.7682)
│   ├── mbert.pt        # bert-base-multilingual-cased (val F1=0.7753)
│   └── kobigbird.pt    # kobigbird-bert-base (val F1=0.7861)
└── docs/
    ├── fusion_architecture.html  # 구조도(단일모델 파이프라인 + 앙상블)
    └── 모델설명.md         # 상세 설명(아래 참고)
```

## 의존성
- `requirements.txt`에 명시된 파이썬 패키지(torch/torchvision/transformers 등).
- HuggingFace 모델: `klue/roberta-large`, `xlm-roberta-large`, `beomi/kcbert-large`,
  `bert-base-multilingual-cased`, `monologg/kobigbird-bert-base`,
  `facebook/dinov3-vitl16-pretrain-lvd1689m` — **최초 실행 시** 자동으로 다운로드/캐시됨
  (인터넷 연결 필요, 총 용량 대략 5~6GB). 두 번째 실행부터는 로컬 캐시 사용.
- GPU 없어도 동작은 하지만(`--device cpu`) 텍스트 백본 5개를 다 CPU로 돌리면 많이 느림.

## 참고
- **텍스트 입력은 description 문장만 사용** — 문양유형/재질/시대 같은 구조적 메타데이터는
  넣지 않음(이전 버전은 메타데이터를 같이 넣었는데, description만 쓰는 게 원래 의도였다는
  게 확인돼서 이번 버전부터 뺐음). 메타데이터를 빼도 성능 차이는 크지 않았음(±0.003~0.004).
- 총 5개 모델(klue, xlmr, kcbert, mbert, kobigbird) — 전부 description-only, end-to-end
  fine-tuning. 모델별 fine-tuning 설정을 소폭 튜닝해서 검증 성능을 확인 후 채택함(자세한
  근거는 `docs/모델설명.md` §5 참고).
- mbert(bert-base-multilingual-cased), kobigbird(monologg/kobigbird-bert-base)는 이번에
  새로 추가한 아키텍처 — 기존 klue/xlmr/kcbert 계열과 사전학습 코퍼스·구조가 달라서
  앙상블 다양성에 기여함. (kobigbird는 sparse attention 구조지만 이번 max_length=512
  설정에서는 sequence가 짧아 자동으로 full attention으로 대체됨 — 그래도 사전학습 자체가
  다른 계열이라 다양성 효과는 있음.)
- 4개 모델(klue_bert 포함) 조합은 0.7991로 0.8 미달이었고, klue_bert를 kobigbird로
  바꾼 5개 조합이 정확히 0.8000을 달성해서 이걸 채택함. 2~6개 전체 조합 탐색 결과이므로
  이보다 적은 개수로는 0.8을 넘기 어려움.
- 이미지는 DINOv3로 frozen 인코딩, 각 텍스트 모델은 end-to-end fine-tuning됨. 대조군으로
  텍스트 인코더를 frozen으로 썼을 때와 비교하면 fine-tuning의 효과가 뚜렷함:

  | 텍스트 인코더 처리 | F1@5 |
  |---|---|
  | Frozen | ~0.68 |
  | Fine-tune (채택) | 0.77~0.79 |
- 앙상블 가중치는 val 446장 기준 랜덤서치로 탐색한 값(config.json의 `ensemble_weight`).
- 상세 실험 과정/정정 이력은 `../train/RESULT.md`, `../train/설명.md` 참고.
- 학습 코드(재현용)는 `../train/` 참고.

## 구조도 & 상세 설명
- [`docs/fusion_architecture.html`](docs/fusion_architecture.html) — 모델 구조도 (단일모델
  파이프라인의 Q=K=V self-attention 상세 + 5개 모델 앙상블 가중합)
- [`docs/모델설명.md`](docs/모델설명.md) — 사용 모델, 학습 디테일, 설계 이유(왜 이미지는
  얼리고 텍스트는 학습시켰는지, 왜 description만 쓰는지, 왜 5개를 앙상블했는지), **learning
  rate를 다르게 잡은 의도**, 그리고 **"얼리거나 같은 LR로 풀면 어떻게 되는지"를 실제로
  실험한 결과**까지 전부 정리.
