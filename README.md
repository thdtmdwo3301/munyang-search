# 전통문양 감성/형용사 분류

전통문양 이미지와 기존 설명문(description)을 입력받아 22개 감성/형용사 중 상위 5개를 출력합니다. DINOv3 이미지 인코더와 텍스트 모델 5종을 결합한 **최종 앙상블 F1@5는 0.8000(80.00%)**입니다.

## 성능 및 실행 환경

| 구분 | 내용 |
|---|---|
| 데이터 | 전체 4,400건 · 학습 3,954건 · 검증 446건, 분할 seed 42 |
| 최종 앙상블 | KLUE · XLM-R · KcBERT · mBERT · KoBigBird |
| 기존 학습 GPU | NVIDIA GeForce RTX 3090 |
| 기존 검증 성능 | F1@5 **0.8000** |
| 재추론 확인 | 2026-09-07 · F1@5 **0.8000** (1,784 / 2,230) |
| 재추론 GPU | NVIDIA GeForce RTX 3090 **1개 (24GB)** |
| 재추론 환경 | Linux · Python 3.11.10 · PyTorch 2.5.1+cu121 · CUDA 12.1 · Transformers 5.13.0 |

재추론은 기존 검증자료 446건의 이미지에서 DINOv3 특징을 새로 추출하고 기존 설명문을 함께 입력해 확인했습니다. 이 자료는 모델·앙상블 가중치 선택에도 사용됐으며, 설명문의 감성어는 제거하지 않았습니다. 독립 시험 또는 이미지 단독 성능이 아닙니다.

## 설치

Python 3.11, NVIDIA GPU 드라이버, [Git LFS](https://git-lfs.com/)를 준비합니다. 아래 설치 파일은 Linux/Windows의 CUDA 12.1 환경용입니다.

```bash
git lfs install
git clone https://github.com/thdtmdwo3301/munyang-search.git
cd munyang-search
git lfs pull
```

가상환경을 생성하고 활성화합니다.

```bash
# Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

저장소 루트에서 라이브러리 설치와 가중치 준비를 실행합니다.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python inference/prepare_weights.py
```

가중치 5개는 Git LFS로 제공하며 총 약 6.24GB입니다. 분할된 XLM-R 가중치는 준비 도구가 `inference/weights/xlmr.pt`로 복원하고 원본 SHA-256을 검증합니다. 최초 추론에는 Hugging Face 모델·토크나이저 캐시 다운로드와 DINOv3 모델 접근 권한이 필요할 수 있습니다.

## 실행

```bash
python inference/infer.py --image /path/to/image.jpg --description "문양의 기존 설명문"
```

### 내 컴퓨터에서 Gradio 실행

위 설치와 가중치 준비를 마친 뒤, 가상환경을 활성화한 상태에서 저장소 루트에서 실행합니다.

```bash
python gradio/app_gradio.py
```

프로그램을 실행한 컴퓨터의 브라우저에서 `http://localhost:7860`에 접속합니다. 이미지를 업로드하고 설명문을 입력하면 감성/형용사 상위 5개를 확인할 수 있습니다. 사용 중에는 실행 터미널을 열어 둡니다.

## 상세 안내

- [추론 및 모델별 성능](inference/README.md)
- [학습 방법 및 실험 설명](train/README.md)
- [기존 실험 결과](train/RESULT.md)
- [내 컴퓨터에서 Gradio 실행](gradio/README.md)
