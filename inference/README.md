# 감성/형용사 추론 — 최종 F1@5 0.8000

이미지(DINOv3)와 기존 설명문(description)을 사용하는 5종 융합 모델의 가중 앙상블입니다. **배포 모델 전체의 F1@5는 0.8000(80.00%)**이며, 아래 개별 모델 점수와 구분합니다.

## 실행

[루트 설치 안내](../README.md#설치)에 따라 라이브러리와 Git LFS 가중치를 준비한 뒤, 저장소 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python inference/prepare_weights.py
python inference/infer.py --image /path/to/image.jpg --description "문양의 기존 설명문"
```

`--description`에는 기존 annotations의 설명문을 입력합니다. 구조적 메타데이터나 annotations-db의 별도 캡션은 사용하지 않습니다. 출력은 감성/형용사 상위 5개와 각 앙상블 점수입니다.

Python에서 사용하려면 저장소 루트에서 다음과 같이 불러옵니다.

```python
from inference.infer import EnsembleModel

model = EnsembleModel()
labels, scores = model.predict(image_path, description, topk=5)
```

## 성능

| 평가 대상 | F1@5 | 앙상블 가중치 |
|---|---:|---:|
| KLUE 개별 융합 모델 | 0.7794 | 0.3135 |
| XLM-R 개별 융합 모델 | 0.7753 | 0.2099 |
| KcBERT 개별 융합 모델 | 0.7682 | 0.2143 |
| mBERT 개별 융합 모델 | 0.7753 | 0.2106 |
| KoBigBird 개별 융합 모델 | 0.7861 | 0.0518 |
| 5종 동일가중 앙상블 | 0.7946 | — |
| **최종 5종 가중 앙상블** | **0.8000** | — |

이전에 표기된 `0.77~0.79`는 개별 모델의 성능 범위입니다. 최종 배포 모델은 5개 모델을 함께 사용하므로 **0.8000**으로 표시합니다. 추론은 `config.json`의 가중치로 확률을 합산하며, 가중치 합 1.0001로 나누는 추가 정규화는 하지 않습니다.

## 재추론 확인 — 2026-09-07

- GPU: **NVIDIA GeForce RTX 3090 1개 (24GB)**.
- 환경: Python 3.11.10, PyTorch 2.5.1+cu121, CUDA 12.1, Transformers 5.13.0.
- 기존 검증자료 446건의 ID·순서·라벨 목록을 백업 검증 파일과 대조했습니다.
- 이미지를 직접 입력해 DINOv3 특징을 새로 추출하고 기존 설명문으로 추론했습니다.
- 정답 일치 1,784개 / 2,230개, **F1@5 0.8000**을 재현했습니다.

기존 검증자료는 모델·가중치 선택에도 사용됐으며 설명문의 감성어는 제거하지 않았습니다. 이 결과는 독립 시험 또는 이미지 단독 성능이 아닙니다.

## 파일 구성

- `infer.py`: 이미지와 설명문을 입력받는 추론 코드.
- `config.json`: 모델 5종, 라벨 목록, 앙상블 가중치.
- `weights/`: 학습된 가중치. XLM-R은 Git LFS 분할 파일에서 복원합니다.
- `prepare_weights.py`, `weights-manifest.json`: 가중치 복원 및 SHA-256 검증.
- [모델 상세 설명](docs/모델설명.md), [구조도](docs/fusion_architecture.html).
- [학습 README](../train/README.md), [기존 실험 결과](../train/RESULT.md).
