# 전통문양 검색·감성분류 공개 소스

문양 이미지와 설명(description)을 이용해 22개 감성 라벨 중 Top-5를 예측하는 연구의 **소스 코드 공개본**입니다. 이 저장소는 기존 F1@5 `0.8000` 기준선 기록의 공개 코드와 학습된 가중치 5개를 제공합니다. 이후 개인연구 실험은 포함하지 않습니다.

## 공개 범위

현재 저장소에는 다음 디렉터리의 코드·설명 문서와 Git LFS로 관리하는 기준선 가중치가 포함되어 있습니다.

```text
train/       학습 코드와 데이터셋 인터페이스
inference/   추론 코드·설정 예시·모델 설명
val/         검증 결과 분석 코드
gradio/      Gradio 데모 코드
```

다음 항목은 포함하지 않습니다.

- 원본·전처리 데이터와 이미지
- 최종 추론 가중치 5개 외의 체크포인트
- DINOv3 특징과 저장된 validation 확률(`*.npz`, `*.npy`)
- 모델 캐시와 실험 결과물

사용자는 권한이 있는 데이터를 준비하고 아래 절차로 Git LFS 가중치를 내려받아야 합니다. 모델 코드와 가중치를 받는 것만으로 데이터까지 제공되는 것은 아닙니다. 학습 데이터 경로는 `MUNYANG_DATA_ROOT` 환경변수로 지정합니다.

## 코드와 학습된 가중치 받기

먼저 [Git LFS](https://git-lfs.com/)를 설치한 뒤 실행합니다.

```bash
git lfs install
git clone https://github.com/thdtmdwo3301/munyang-search.git
cd munyang-search
git lfs pull
python inference/prepare_weights.py
```

이미 저장소를 받은 경우 해당 폴더에서 `git pull`, `git lfs pull`, `python inference/prepare_weights.py`를 실행합니다. GitHub의 일반 ZIP 다운로드 대신 Git LFS 방식으로 받으세요.

가중치 총 크기는 약 6.24 GB입니다. GitHub Free/Pro의 파일당 2 GB 제한에 맞춰 `xlmr.pt`만 `xlmr.pt.part1`, `xlmr.pt.part2`로 나눠 저장합니다. 준비 도구가 원래 `xlmr.pt`로 복원하며, 모든 가중치의 SHA-256을 백업 원본과 대조합니다. 재학습이나 가중치 변환은 하지 않습니다. 기존 파일이 다르면 덮어쓰지 않고 중단합니다.

가중치는 `2026-08-11_image_text_08` 백업의 최종 5종 앙상블입니다. **F1@5 0.8000은 모델·가중치 선택에 사용한 기존 검증자료 446건의 기록**입니다. 이미지와 기존 description을 함께 사용한 값이며, 독립 시험·이미지 단독·새 데이터의 성능을 뜻하지 않습니다. 이번 배포 준비에서는 재추론·재학습을 수행하지 않았습니다.

`train/ensemble.py`는 저장된 검증 예측값으로 점수를 재계산합니다. 해당 예측값은 이번 가중치 배포에 포함하지 않으며, 가중치만 내려받아서 이 평가 스크립트를 바로 실행할 수는 없습니다. 추론은 아래 `inference/infer.py`를 사용합니다.

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

준비 도구 실행 후 아래 다섯 체크포인트가 구성됩니다. 실제 추론에는 Hugging Face 모델·토크나이저 캐시도 필요하며, 최초 실행 시 다운로드와 해당 모델 접근 권한이 필요할 수 있습니다.

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
