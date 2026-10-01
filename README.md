# 전통문양 감성/형용사 분류

전통문양 이미지와 원설명문을 함께 입력해 22개 감성/형용사 중 상위 5개를 예측합니다. DINOv3 이미지 인코더와 텍스트 융합 모델 5종(KLUE, XLM-R, KcBERT, mBERT, KoBigBird)의 **최종 앙상블 F1@5는 0.8000(80.00%)**입니다.

`run.sh` 또는 `run_windows.bat`을 실행하면 DINOv3 이미지 특징을 새로 추출해 실제 5종 앙상블 추론을 수행하고, 아래 내용을 담은 정적 평가 화면 `report.html`을 생성합니다.

- 정답 일치 수준별 사례 4개(5/5, 4/5, 3/5, 2/5 이하)
- 문양 이미지와 이미지 아래의 원설명문
- 맞힌 정답, 놓친 정답, 오답
- 정답 5개와 앙상블 Top-5 점수
- 실행할 때마다 달라지는 사례 조합과 재현 가능한 seed

Gradio나 별도 웹 서버는 사용하지 않습니다. 결과 HTML은 이미지까지 포함된 단일 파일이라 그대로 전달하거나 브라우저에서 열 수 있습니다.

## 성능 및 실행 환경

| 구분 | 내용 |
|---|---|
| 데이터 | 전체 4,400건 · 학습 3,954건 · 검증 446건, 분할 seed 42 |
| 최종 앙상블 | KLUE · XLM-R · KcBERT · mBERT · KoBigBird |
| 기존 검증 성능 | F1@5 **0.8000** (1,784 / 2,230) |
| 확인 GPU | NVIDIA GeForce RTX 3090 1개(24GB) |
| 확인 환경 | Linux · Python 3.11.10 · PyTorch 2.5.1+cu121 · CUDA 12.1 |

기존 검증은 검증자료 446건의 이미지에서 DINOv3 특징을 새로 추출하고 원설명문을 함께 입력해 확인했습니다. 검증자료는 모델 및 앙상블 가중치 선택에도 사용됐고 설명문의 감성어는 제거하지 않았으므로, 독립 시험 또는 이미지 단독 성능은 아닙니다.

## GitHub에서 받기

Python 3.11, NVIDIA GPU 드라이버와 [Git LFS](https://git-lfs.com/)를 먼저 준비합니다.

```bash
git lfs install
git clone https://github.com/thdtmdwo3301/munyang-search.git
cd munyang-search
git lfs pull
```

가중치 5개는 Git LFS로 제공하며 총 약 6.24GB입니다. 최초 실행 시 Hugging Face의 DINOv3와 텍스트 모델도 내려받으므로 인터넷 연결과 충분한 저장공간이 필요합니다.

## Linux 실행

저장소 루트에서 다음 두 명령을 순서대로 실행합니다.

```bash
bash setup.sh
bash run.sh
```

결과는 `outputs/static_eval_날짜_시간_식별자/report.html`에 생성됩니다. GUI가 있는 환경에서는 기본 브라우저도 함께 열립니다. GUI가 없는 Linux 서버에서는 출력된 경로의 HTML 파일을 내려받아 확인하면 됩니다.

## Windows 실행

가장 간단한 방법은 저장소의 `START_WINDOWS.bat`을 더블클릭하는 것입니다. 처음에는 설치까지 진행하고, 이후에는 바로 평가를 실행합니다.

명령 프롬프트에서는 아래와 같이 나눠 실행할 수 있습니다.

```bat
setup_windows.bat
run_windows.bat
```

결과 HTML이 생성된 뒤 기본 브라우저로 열립니다.

## 같은 사례 재현하기

일반 실행은 직전 실행과 다른 4개 사례 조합을 선택합니다. 결과 폴더의 `results.json`에 seed가 기록됩니다. 같은 조합을 다시 실행하려면 다음처럼 seed를 지정합니다.

```bash
# Linux
EVALUATION_SEED=12345 bash run.sh
```

```bat
rem Windows
set EVALUATION_SEED=12345
run_windows.bat
```

## 이미지 한 장 직접 추론하기

```bash
python inference/infer.py --image /path/to/image.jpg --description "문양의 원설명문"
```

## 상세 안내

- [추론 및 모델별 성능](inference/README.md)
- [학습 방법 및 실험 설명](train/README.md)
- [기존 실험 결과](train/RESULT.md)
