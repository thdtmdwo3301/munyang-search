# 문양 감성분류 프로그램

이미지 단독 및 이미지·원설명문 입력의 감성어 22개 중 상위 5개를 예측한다.
평가, HTML 시각화, 입력별 결과 조회, 공통 모듈 재학습을 제공한다.
Docker는 필수가 아니다.

## 지원 환경

검증 대상은 Linux x86_64, Python 3.11, NVIDIA CUDA GPU이다.
현재 평가 설정은 BF16을 사용하는 Ampere 이상 GPU를 요구한다. 검증 장비는 RTX 3090 24GB이다.
NVIDIA 드라이버는 호스트에 설치되어 있어야 한다. PyTorch의 CUDA 실행 라이브러리는 requirements.txt로 설치한다.
CPU는 결과 조회·HTML 생성에 사용할 수 있으나 전체 평가의 동등 성능/속도는 검증하지 않았다.
Windows에서는 WSL2/Linux 환경을 사용한다. 다른 OS/GPU에서의 실행은 별도 확인이 필요하다.

## 설치

GitHub에서 코드를 받는 경우 저장소 주소를 실제 주소로 바꾼다.
코드 ZIP을 풀어서 사용할 수도 있다.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/thdtmdwo3301/munyang-search.git pattern-search
cd pattern-search
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

python3은 Python 3.11이어야 한다. 또는 PYTHON_BIN=python3.11 bash setup.sh를 실행한다.
venv는 다른 시스템으로 복사하지 않고 대상 시스템에서 새로 만든다.
패키지 설치에는 인터넷이 필요하다. 설치 후 포함된 모델의 추론에는 Hugging Face 접속이나 인증이 필요하지 않다.

## 가중치·데이터 배치

USB 전체 묶음에는 아래 assets/와 data/가 포함된다.
코드 묶음에는 텍스트 모델 설정·토크나이저와 검증 대응 목록만 포함한다. 가중치, 원본 이미지와 주석은 기존 전달 자료에서 연결한다.
기존 연구 데이터와 가중치의 공유 범위는 그대로 따른다. 자동 업로드나 외부 전송 기능은 없다.

```text
pattern-search/
  main.py
  run.sh / setup.sh / run_train.sh
  requirements.txt / requirements-lock-linux-py311.txt
  configs/                 # 모델·배포 설정
  evct/                    # 특징 추출·분류·학습·시각화
  experiments/             # 실제 평가 실행
  assets/
    weights/               # 이미지 2종, BERT 융합 5종, 메모리 가중치
    dinov3l/               # 이미지 모델
    text/                  # 5종 텍스트 모델 설정·토크나이저
  data/
    annotations/
    images/
  outputs/                 # 실행별 결과
```

프로젝트 내부 경로를 기본으로 사용한다. 다른 위치에 자산을 두려면 configs/deployment.json을 수정한다.
이 파일의 상대 경로 기준은 프로젝트 루트다. configs/single_*.json의 상대 경로 기준은 해당 JSON 파일의 폴더다.
서버명, Docker 이름, 특정 SSD 장치명, 사용자 홈 경로를 코드에 지정하지 않는다.

## 평가와 시각화 — 한 번에 실행

```bash
bash run.sh
```

환경·입력 확인 → 원본 이미지에서 실제 추론 → F1@5 계산 → HTML 결과 생성 순서로 실행한다.
기본값은 이미지 단독과 이미지·원설명문 두 방식, batch 24, 검증 446건이다.
이미 저장된 예측을 실제 추론의 입력으로 재사용하지 않는다.
실행마다 새 outputs/날짜_실행ID/를 만들며 기존 결과를 덮어쓰지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh
bash run.sh --mode image
bash run.sh --mode multimodal
bash run.sh --memory off
bash run.sh --open
```

--open은 실행한 시스템에서 브라우저를 연다. SSH 서버에서는 생성된 report.html을 PC로 복사해 열면 된다.
HTML은 이미지가 내장되어 있어 별도 웹 서버가 필요 없다. 입력 방식별 필터 버튼과 정답/오답 사례를 포함한다.

결과 파일:
- results.json: F1@5, 처리 건수, 실행 조건
- image_predictions.jsonl / multimodal_predictions.jsonl: 문양별 상위 5개 예측
- end_to_end_predictions.npz: 실제 예측 점수 배열
- report.html: 정량 그래프·정성 사례
- outputs/latest.json: 가장 최근 완료된 평가 위치

### 평가 조건

기본 memory=released는 기존 평가 조건을 보존한다. 검증 446건 중 178건(40%)의 정답 라벨 메모리를 적용한다.
따라서 이 결과를 미사용 독립 검증 성능으로 해석하지 않는다.
--memory off는 해당 보정을 끈 별도 평가다. 이미지·텍스트 입력의 텍스트는 원설명문이며 자동 생성 캡션이 아니다.
환경 차이에 따라 수치가 달라질 수 있으므로 실제 출력값을 사용한다.

## 결과 조회 / 시험절차서 화면

아래 명령은 환경 및 저장된 결과 조회다. 전체 재추론은 bash run.sh로 실행한다.
```bash
python main.py --step environment
python main.py --step models
python main.py --step image
python main.py --step multimodal
python main.py --step prediction
python main.py --step metrics
```

첫 평가를 완료한 뒤 결과 조회를 실행한다. 이전 실행을 지정하려면 --result-dir outputs/실행ID를 추가한다.
HTML만 다시 만들려면 python main.py report를 실행한다.
환경·파일 준비만 확인하려면 python main.py check를 실행한다.

## 모듈 교체 / 단일 모델 추론

이미지 특징 모듈, 텍스트 특징 모듈, 분류기 모듈은 evct/model.py의 공통 연결을 사용한다.
configs/modules.json은 기존 평가 모델의 모듈 설정이다.
새 클래스는 package.module:Class 형식으로 지정할 수 있다.
모듈의 입력 차원과 가중치가 일치해야 한다. 구조를 바꾼 경우 해당 구조로 학습된 가중치가 필요하다.

```bash
python -m evct.predict --config configs/single_klue.json \
  --mode multimodal --image data/images/이미지파일.jpg --text "문양 설명문"
```

이 명령은 KLUE 한 모델 추론으로, 5종 앙상블이나 정답 메모리 보정을 적용하지 않는다.
image/text/multimodal 모드를 지원하되 해당 입력 방식에 맞는 설정·학습 가중치를 지정해야 한다.
현재 전달 가중치에는 텍스트 단독으로 학습된 분류기가 포함되지 않는다.

## 공통 모듈 재학습

```bash
bash run_train.sh --config configs/single_klue.json \
  --data-root data --output outputs/retrain_new \
  --epochs 3 --batch-size 4 --lr 0.00001
```

추론과 동일한 모듈 연결과 체크포인트 로더를 사용한다.
BCEWithLogitsLoss와 AdamW로 학습하며, 기존 전체 학습 실험을 그대로 재현하는 명령은 아니다.
--freeze-text / --freeze-image는 해당 인코더를 고정한다.
feature_image_encoder로 지정한 외부 이미지 특징 추출기는 항상 고정된다.
이미지 인코더도 학습하려면 configs/single_image.json처럼 model.image_encoder에 포함한다.

변경 데이터는 --data-root 대신 --train-jsonl 파일 --val-jsonl 파일로 지정한다.
각 행: id, image_path, description, emotions(정답 감성어 5개).
사용하지 않는 입력 필드는 생략 가능하다. 상대 image_path는 JSONL 파일 폴더 기준이다.
학습/검증 ID 중복, GT 오류, 필수 입력 누락을 차단한다. 정답 메모리를 학습·검증에 사용하지 않는다.

결과: model.pt(마지막 epoch), predict.json, run.json, predictions.jsonl, verification.json.
저장 후 새 모델로 다시 불러와 검증 점수의 일치를 확인한다.
predict.json을 evct.predict의 --config로 전달하면 저장된 모델로 추론할 수 있다.
이 파일을 재학습 --config로 주면 가중치부터 이어 학습한다. optimizer/RNG의 정확한 중단 지점 재개는 아니다.

## 배포 검증

venv는 include-system-site-packages=false로 설치하여 Docker 사전 설치 패키지를 공유하지 않는다.
배포 검증 기록은 Drive 코드 ZIP의 PORTABILITY_VERIFICATION.json을 참고한다.
requirements-lock-linux-py311.txt는 검증 환경의 전체 버전을 고정한다.
정확한 패키지 재설치는 pip install -r requirements-lock-linux-py311.txt로 할 수 있다.

코드 ZIP은 코드·설정·문서와 실행용 토크나이저·검증 대응 목록을 포함한다. USB용 ZIP은 코드·모델·데이터를 포함하며 venv와 캐시는 제외한다.
물리 USB 복사는 별도로 진행한다.

기본 예측 사례: --step prediction은 KC_TP_MJ_LA_M002642_00을 표시한다. 전체 데이터의 첫 행이 아닌 지정된 시연 사례이며, --case-id로 다른 문양을 선택할 수 있다. 전체 평가 지표에는 영향을 주지 않는다.

TC3 개별 캡처: --step image와 --step multimodal은 해당 입력 방식의 처리 건수와 F1@5를 함께 표시한다. 성능만 분리 조회하려면 --step metrics --mode image 또는 --step metrics --mode multimodal을 사용한다. --step metrics의 기본값은 두 방식 모두 표시다.


이미지·설명문 대응 확인: data/evaluation_manifest.jsonl은 원본 주석과 seed=42 검증 분할로 작성한 고정 446건 목록이다. 외부 시험기관이 별도 제공한 목록은 아니다. 평가 시 실제 입력의 ID, 이미지 상대 경로, 파일 SHA-256, 설명문 원문을 추론 전에 비교하며 불일치 시 중단한다. input_check.json과 input_pairs.jsonl에 검사 및 실제 입력을 저장한다. --step multimodal은 저장 기록과 현재 파일을 재확인하여 대응 검사 결과를 표시한다. 검사 기록이 없는 과거 실행에는 PASS를 표시하지 않는다.


## 최신 코드와 가중치 설치

Drive 폴더에는 최신 코드 ZIP과 assets/ 가중치 폴더를 함께 제공한다.
1. pattern-search-code_20261006.zip을 풀면 pattern-search/ 폴더가 생성된다.
2. Drive의 assets/weights와 assets/dinov3l을 각각 pattern-search/assets/weights,
   pattern-search/assets/dinov3l에 넣는다. ZIP에 있는 assets/text는 그대로 유지한다.
3. 기존 데이터의 images/, annotations/를 pattern-search/data에 넣는다.
   ZIP에 포함된 evaluation_manifest.jsonl과 evaluation_manifest_info.json은 유지한다.
4. Python 3.11로 가상환경을 생성하고 requirements.txt를 설치한 뒤 bash run.sh를 실행한다.

기본 위치로 배치하면 가중치 경로를 변경하지 않아도 된다.
가중치·데이터를 외부 폴더에 둘 경우 configs/deployment.json의 weights, image_model,
data를 변경한다. 절대 경로 및 프로젝트 루트 기준 상대 경로를 지원한다.
data를 변경하면 고정 검증 대응 목록 두 파일도 그 데이터 폴더에 둔다.
단일 모델 추론·재학습 경로는 별도 configs/single_*.json에서 checkpoint,
image_processor, feature_image_encoder.model_name 등을 변경한다.
이 설정과 configs/ensemble.json의 상대 경로는 JSON 파일 폴더 기준이다.
텍스트 설정·토크나이저 위치를 옮기면 ensemble의 models[].text_model도 변경한다.
설정 후 python main.py check로 확인한다.
원본 데이터, venv, 실행 결과 및 캐시는 이번 Drive 배포에 포함하지 않는다.

## GitHub에서 설치할 때 추가 자료

[코드·가중치 배포 폴더](https://drive.google.com/drive/folders/1EhuOVDmiHiUNQ-pW-0CeBS94jOIDBu6y)에서 자료를 받는다.
GitHub에는 원본 연구 데이터, 평가 실행 결과, 가중치를 새로 업로드하지 않는다.
clone 후 Drive의 코드 ZIP에서 assets/text/, data/evaluation_manifest.jsonl,
data/evaluation_manifest_info.json, ASSET_MANIFEST.json을 같은 상대 경로로 복사한다.
Drive의 assets/weights와 assets/dinov3l도 저장소의 assets/ 아래에 넣는다.
원본 images/와 annotations/는 별도로 제공받은 데이터를 사용한다.
기존 Git LFS 가중치 포인터는 이전 실험 자료로 유지한다. 최신 실행은 Drive 가중치를 사용하므로
위 clone 명령은 기존 LFS 가중치의 자동 다운로드를 생략한다.

## 이전 실험 자료

inference/, evaluation/, val/, train/README.md, train/RESULT.md와 Windows 배치 파일은
이전 5종 앙상블 실험 자료다. 현재 배포본의 기본 실행은 Linux/WSL2에서 main.py와 run.sh를 사용한다.
이전 실험 설명의 모델 구성·점수와 현재 평가 조건을 구분한다.
