# 전통문양 검색·감성분류 공개 소스

문양 이미지와 설명(description)을 이용해 22개 감성 라벨 중 Top-5를 예측하는 연구의 **소스 코드 공개본**입니다. 이 저장소는 기존 F1@5 `0.8000` 기준선 기록을 바탕으로 정리했지만, 최신 실험 전체본이나 공식 릴리스는 아닙니다.

## 공개 범위

현재 저장소에는 다음 디렉터리의 코드와 설명 문서만 포함되어 있습니다.

```text
train/       학습 코드와 데이터셋 인터페이스
inference/   추론 코드·설정 예시·모델 설명
val/         검증 결과 분석 코드
gradio/      Gradio 데모 코드
```

다음 항목은 포함하지 않습니다.

- 원본·전처리 데이터와 이미지
- 체크포인트·가중치(`*.pt`, `*.pth`, `*.ckpt`)
- DINOv3 특징과 저장된 validation 확률(`*.npz`, `*.npy`)
- 모델 캐시와 실험 결과물

따라서 사용자는 권한이 있는 데이터와 모델 파일을 별도로 준비해야 하며, 공개 저장소만으로 학습·추론을 즉시 실행할 수 없습니다. 학습 데이터 경로는 `MUNYANG_DATA_ROOT` 환경변수로 지정합니다.

## 설치·실행 안내

권장 환경은 Python 3.11이며, 기준 실행 환경은 PyTorch 2.5.1/CUDA 12.1입니다.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r inference/requirements.txt -r gradio/requirements.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r inference/requirements.txt -r gradio/requirements.txt
```

실제 추론에는 아래 다섯 체크포인트와 Hugging Face 모델 캐시가 필요합니다.

```text
inference/weights/
├── klue.pt
├── xlmr.pt
├── kcbert.pt
├── mbert.pt
└── kobigbird.pt
```

자산을 준비한 뒤 저장소 루트에서 실행합니다.

```bash
python inference/infer.py --image /path/to/image.jpg --description "문양 설명"
python gradio/app_gradio.py
```

학습 데이터는 다음 구조로 준비하고 `MUNYANG_DATA_ROOT`로 위치를 지정합니다.

```text
<MUNYANG_DATA_ROOT>/
├── annotations/
├── images/
└── annotations-db/
```

```powershell
$env:MUNYANG_DATA_ROOT="D:\data\munyang"
```

```bash
export MUNYANG_DATA_ROOT=/path/to/munyang
```

학습 코드는 `train/`에 있으며 미리 추출한
`train/dinov3_features/features_train.pt`와 `features_val.pt`를 요구합니다. 학습 결과 확률을
앙상블하려면 저장소 루트에서 `cd train` 후 `python ensemble.py`를 실행합니다. 세부 추론 구조는
`inference/README.md`와 `inference/docs/`, 데모 옵션은 `gradio/README.md`를 참고하세요.

## 주의

서버 주소, 계정, 비밀번호, 토큰과 절대 경로는 공개하지 않습니다. 연구 재현 시 데이터·모델의 사용 권한과 해당 모델의 라이선스를 먼저 확인하세요.
