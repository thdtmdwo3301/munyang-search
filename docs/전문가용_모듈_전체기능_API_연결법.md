# 전문가용 모듈 전체 기능 및 API 연결 설명서

- 대상 시스템: 문양 감성분류 프로그램(eVCT)
- 기준 버전: GitHub `main`, 2026-10-08
- 대상 독자: 시스템 운영자, 개발자, API 연동 담당자

## 1. 시스템 개요

이 프로그램은 전통문양 이미지 또는 이미지와 원설명문을 입력받아 22개 감성어 중 점수가 높은 5개를 반환한다. 기본 평가는 다음 두 입력 방식을 지원한다.

| 방식 | 입력 | 모델 구성 |
|---|---|---|
| 이미지 단독 | 문양 이미지 | DINOv3-L 이미지 인코더 + 이미지 MLP 분류기 |
| 멀티모달 | 문양 이미지 + 원설명문 | DINOv3-L 특징 + BERT 계열 5종 텍스트 모델 + Attention 융합 + 가중 앙상블 |

텍스트 모델 5종은 KLUE-RoBERTa, XLM-R, KcBERT, mBERT, KoBigBird다. 기본 멀티모달 앙상블 가중치는 `configs/ensemble.json`에 기록되어 있다.

> 기존 F1@5 80.00%는 고정 검증자료에 대한 기존 평가값이다. 기본 `memory=released` 평가는 446건 중 178건에 정답 라벨 메모리를 적용하므로 미사용 독립 검증 성능으로 해석하면 안 된다. 독립적인 비교가 필요하면 `--memory off`를 사용하고 새 시험자료로 평가한다.

## 2. 제공 기능

| 기능 | 실행 방법 | 주요 결과 |
|---|---|---|
| 환경·자산 확인 | `python main.py check` | Python, GPU, 모델, 데이터 유효성 확인 |
| 전체 평가 | `bash run.sh` | 실제 추론, F1@5, 예측 파일, HTML 보고서 |
| 입력 방식별 평가 | `bash run.sh --mode image` 등 | 이미지 단독 또는 멀티모달 결과 |
| 보고서 재생성 | `python main.py report` | 저장 결과로 `report.html` 재생성 |
| 결과 조회 | `python main.py --step ...` | 환경·모델·예측·성능을 터미널에 표시 |
| 단일 모델 추론 | `python -m evct.predict ...` | JSON 형식 Top-5 결과 |
| 모듈 재학습 | `bash run_train.sh ...` | 체크포인트, 추론 설정, 검증 결과 |
| 모듈 교체 | JSON 설정의 `type` 변경 | 내장 또는 사용자 정의 PyTorch 모듈 로드 |
| 자산 검증 | `python tools/prepare_assets.py --verify` | 가중치 복원 및 SHA-256 확인 |
| 배포 묶음 생성 | `python tools/export_bundle.py ...` | 코드용/USB용 ZIP과 해시 목록 |

## 3. 지원 환경과 설치

검증 환경은 Linux x86_64, Python 3.11, NVIDIA CUDA GPU다. 기본 전체 평가는 BF16을 사용하므로 Ampere 이상 GPU가 필요하며 검증 장비는 RTX 3090 24GB다. Windows에서는 WSL2/Linux 사용을 권장한다.

```bash
git lfs install
git clone https://github.com/thdtmdwo3301/munyang-search.git pattern-search
cd pattern-search
git lfs pull --include="assets/**" --exclude=""
bash setup.sh
source venv/bin/activate
```

수동 설치도 가능하다.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

설치 후 다음 명령으로 자산과 실행 환경을 확인한다.

```bash
python tools/prepare_assets.py --verify
python main.py check
```

`main.py check`는 원본 데이터까지 검사하므로 `data/annotations`와 `data/images`가 먼저 준비돼 있어야 한다.

## 4. 주요 디렉터리

```text
pattern-search/
├── main.py                         # 평가·보고서·점검·조회·학습 진입점
├── run.sh / run_train.sh           # 평가 및 학습 실행 스크립트
├── configs/
│   ├── deployment.json             # 데이터·가중치·GPU·출력 경로
│   ├── modules.json                # 평가 모듈 종류와 주요 하이퍼파라미터
│   ├── ensemble.json               # 5종 모델·라벨·앙상블 가중치
│   ├── single_image.json           # 이미지 단독 모델 설정
│   └── single_klue.json            # KLUE 멀티모달 단일 모델 설정
├── evct/                            # 모듈 생성·체크포인트·추론·학습·보고서
├── experiments/                     # 고정 평가 실행 코드
├── tools/                           # 자산 준비·검증·배포 묶음 생성
├── assets/                          # 로컬 모델·토크나이저·가중치
├── data/                            # 외부 제공 이미지·주석
└── outputs/                         # 실행별 결과(기존 결과를 덮어쓰지 않음)
```

상대 경로의 기준은 다음과 같다.

- `configs/deployment.json`: 프로젝트 루트 기준
- `configs/single_*.json`: 해당 JSON 파일이 위치한 폴더 기준
- JSONL의 `image_path`: JSONL 파일이 위치한 폴더 기준

## 5. 전체 평가와 결과 조회

### 5.1 기본 실행

```bash
bash run.sh
```

처리 순서는 자산 복원 → 환경·입력 검증 → 원본 이미지 실제 추론 → F1@5 계산 → HTML 생성이다. 기본값은 이미지 단독과 멀티모달 모두, batch 24, `cuda:0`, `memory=released`다.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
bash run.sh --mode image
bash run.sh --mode multimodal
bash run.sh --memory off
bash run.sh --batch-size 8 --device cuda:1
bash run.sh --open
```

주요 출력은 다음과 같다.

- `outputs/<실행ID>/results.json`: 실행 조건, 처리 건수, F1@5
- `image_predictions.jsonl`: 이미지 단독 Top-5
- `multimodal_predictions.jsonl`: 이미지+설명문 Top-5
- `end_to_end_predictions.npz`: 전체 예측 점수
- `input_check.json`, `input_pairs.jsonl`: 이미지·설명문 대응 검사 기록
- `report.html`: 정량 결과와 정성 사례가 포함된 단일 HTML
- `outputs/latest.json`: 최근 완료 실행 위치

### 5.2 저장 결과 조회

```bash
python main.py --step environment
python main.py --step models
python main.py --step image
python main.py --step multimodal
python main.py --step prediction
python main.py --step prediction --case-id KC_TP_MJ_LA_M002642_00
python main.py --step metrics --mode both
```

특정 실행을 지정하려면 `--result-dir outputs/<실행ID>`를 추가한다. HTML만 다시 만들 때는 다음을 사용한다.

```bash
python main.py report --result-dir outputs/<실행ID> --open
```

## 6. API 연결법

### 6.1 지원 API 범위

현재 배포본은 HTTP/REST 서버를 자동으로 실행하지 않는다. 공식 연결 방식은 다음 두 가지다.

1. 다른 프로그램이 명령행 프로세스를 호출하고 JSON 결과를 읽는 방식
2. 같은 Python 프로세스에서 `evct` 모듈을 불러오는 방식

외부 HTTP API가 필요하면 서비스 계층에서 모델을 한 번 로드한 뒤 위 Python 모듈을 감싸야 한다. 모델을 요청마다 새로 로드하는 방식은 지연시간과 GPU 메모리 사용량 때문에 권장하지 않는다.

현재 `evct.predict`는 설정으로 선택한 **단일 모델**의 한 건 추론 API다. 5종 앙상블 전체는 `main.py evaluate --mode multimodal`의 배치 평가 경로로 제공된다. 따라서 단일 모델 API의 결과를 기존 5종 앙상블 F1@5 80.00%와 동일한 결과로 표시하면 안 된다. 한 건 단위 5종 앙상블 HTTP 서비스가 필요하면 별도 서비스 어댑터와 그에 대한 재검증이 필요하다.

### 6.2 명령행 JSON API

멀티모달 KLUE 단일 모델 예시:

```bash
python -m evct.predict \
  --config configs/single_klue.json \
  --mode multimodal \
  --image data/images/KC_SAMPLE.jpg \
  --text "조선시대 문양의 원설명문"
```

이미지 단독 모델 예시:

```bash
python -m evct.predict \
  --config configs/single_image.json \
  --mode image \
  --image data/images/KC_SAMPLE.jpg
```

정상 출력 형식:

```json
{
  "mode": "multimodal",
  "model_config": "configs/single_klue.json",
  "memory": "off",
  "top5": [
    {"label": "고전적인", "score": 0.91},
    {"label": "장식적인", "score": 0.84}
  ]
}
```

실제 `top5`에는 최대 5개 항목이 점수 내림차순으로 포함된다. 이 단일 모델 API에는 5종 앙상블과 정답 메모리 보정이 적용되지 않는다.

다른 Python 서비스에서 호출하는 예시:

```python
import json
import subprocess

command = [
    "venv/bin/python", "-m", "evct.predict",
    "--config", "configs/single_klue.json",
    "--mode", "multimodal",
    "--image", "data/images/KC_SAMPLE.jpg",
    "--text", "문양의 원설명문",
]
completed = subprocess.run(command, check=True, capture_output=True, text=True)
result = json.loads(completed.stdout)
print(result["top5"])
```

사용자 입력을 연결할 때는 문자열로 셸 명령을 조합하지 말고 위와 같이 인자 배열을 전달한다. 실패 시 프로세스 종료코드는 0이 아니며 오류 내용은 표준 오류에서 확인한다.

### 6.3 Python 모듈 API

핵심 공개 객체는 다음과 같다.

```python
from evct import build_model, build_component, load_checkpoint
from evct.paths import load_model_config
```

| 객체 | 역할 |
|---|---|
| `load_model_config(path)` | 설정 파일을 읽고 로컬 상대 경로를 절대 경로로 변환 |
| `build_model(spec)` | 이미지·텍스트 인코더와 분류기를 조합 |
| `build_component(spec)` | 내장 또는 사용자 정의 모듈 한 개 생성 |
| `load_checkpoint(model, path, layout)` | 체크포인트를 엄격하게 로드 |

모델의 `forward` 입력은 `pixel_values`, `input_ids`·`attention_mask`, 또는 미리 계산한 `image_features`·`text_features`다. 출력은 `[batch, 22]` 형태의 logits이며 확률이 필요하면 `torch.sigmoid(logits)`를 적용한다.

전처리, 토크나이징, BF16 autocast와 Top-5 변환까지 포함한 기준 구현은 `evct/predict.py`다. 장기 실행 서비스는 이 파일의 로딩 절차를 서비스 시작 시 한 번 수행하고, 요청 시 전처리와 forward만 수행하도록 구성한다.

### 6.4 사용자 정의 모듈 연결

설정의 `type`에는 내장 이름 또는 `package.module:Class`를 지정할 수 있다.

내장 이름:

- `dino`: DINO 이미지 인코더
- `hf_text`: Hugging Face 텍스트 인코더
- `image_mlp`: 이미지 단독 분류기
- `text_mlp`: 텍스트 단독 분류기
- `attention_fusion`: 이미지·텍스트 융합 분류기

사용자 정의 인코더는 `torch.nn.Module`을 상속하고 `output_dim` 속성을 제공해야 한다. 분류기는 `required_modalities`를 선언하고 `[batch, labels]` logits를 반환해야 한다.

```json
{
  "type": "my_package.my_module:CustomImageEncoder",
  "model_path": "models/custom"
}
```

입력 차원, 라벨 수, 체크포인트 구조가 모두 일치해야 하며 체크포인트는 `strict=True`로 로드된다. 불일치 항목을 조용히 무시하지 않는다.

## 7. 설정 파일 운영

`configs/deployment.json`에서 다음 값을 변경할 수 있다.

| 키 | 의미 | 기본값 |
|---|---|---|
| `data` | 원본 데이터 루트 | `data` |
| `weights` | 평가 가중치 폴더 | `assets/weights` |
| `image_model` | DINOv3 모델 폴더 | `assets/dinov3l` |
| `ensemble` | 5종 앙상블 설정 | `configs/ensemble.json` |
| `modules` | 모듈 구성 | `configs/modules.json` |
| `outputs` | 결과 저장 폴더 | `outputs` |
| `batch_size` | 평가 배치 크기 | `24` |
| `device` | 실행 장치 | `cuda:0` |
| `memory` | 정답 메모리 조건 | `released` |

운영 설정을 별도 파일로 복사한 뒤 `python main.py evaluate --config <파일>` 형태로 전달할 수 있다. 배치 크기, 장치와 memory는 명령행 옵션이 설정 파일보다 우선한다.

## 8. 자산과 배포

Git LFS 파일이 포인터 상태면 모델이 실행되지 않는다.

```bash
git lfs pull --include="assets/**" --exclude=""
python tools/prepare_assets.py --verify
```

XLM-R 가중치는 `xlmr.pt.part1`, `xlmr.pt.part2`에서 자동 복원되며 크기와 SHA-256이 모두 일치해야 `xlmr.pt`로 게시된다.

배포 묶음 생성 예시:

```bash
# 프로젝트 밖의 빈 폴더를 지정한다.
python tools/export_bundle.py --output ../delivery
python tools/export_bundle.py --output ../delivery-code --code-only
```

가상환경, 캐시, 실행 결과와 인증정보는 배포 ZIP에 포함하지 않는다.

## 9. 운영 시 주의사항

- 프로세스마다 모델을 중복 로드하면 GPU 메모리도 중복 사용된다.
- 기본 전체 평가는 Ampere 이상 CUDA GPU가 필요하다.
- 배치 처리 중 메모리가 부족하면 `--batch-size`를 낮춘다.
- `results.json`의 실행 조건과 `memory` 값을 결과와 함께 보관한다.
- 이미지와 설명문의 대응 검사가 실패하면 추론을 계속하지 말고 원본 목록을 수정한다.
- 새로 재학습한 단일 모델은 `predict.json`으로 먼저 검증한다. 기존 5종 앙상블에 자동 편입되지 않는다.
- 프로그램에는 외부 업로드나 네트워크 전송 기능이 없다. 파일 접근 권한과 API 인증은 연동 서비스에서 관리한다.

## 10. 장애 확인 순서

1. `python --version`이 3.11인지 확인한다.
2. `git lfs pull --include="assets/**" --exclude=""`을 다시 실행한다.
3. `python tools/prepare_assets.py --verify`로 모델 파일을 확인한다.
4. `python main.py --step environment`로 CUDA 인식을 확인한다.
5. `python main.py check`로 데이터·모델·GPU를 함께 확인한다.
6. GPU 메모리 부족이면 배치 크기를 낮춘다.
7. 실행별 `outputs/<실행ID>`와 터미널 오류를 함께 보관한다.
