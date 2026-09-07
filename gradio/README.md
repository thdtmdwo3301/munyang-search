# 내 컴퓨터에서 Gradio 실행

## 감성/형용사 분류 화면 실행
이미지 + description을 넣으면 top-5 감성 라벨/점수를 보여준다. `../inference/`의 config.json과
weights/를 그대로 불러와 쓰므로(가중치 중복 보관 안 함) 실행 전 `../inference/`가 있어야 한다.

[루트 설치 안내](../README.md#설치)에 따라 코드와 가중치를 내려받고 가상환경을 활성화합니다. 아래 명령은 내려받은 저장소의 루트에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python inference/prepare_weights.py
python gradio/app_gradio.py
```

다섯 체크포인트(`klue.pt`, `xlmr.pt`, `kcbert.pt`, `mbert.pt`, `kobigbird.pt`)는
`inference/weights/`에 둡니다. 최초 실행에는 설정에 지정된 DINOv3와 텍스트 모델 캐시도
필요합니다.

- 데이터셋에 있는 이미지(파일명이 원본 그대로인 경우)를 올리면 description을 자동으로
  채워주는 기능이 있는데, 이건 `../train/dataset.py`가 참조하는 원본 ETRI 데이터셋이 로컬에
  있어야 동작함. 없어도 모델 추론 자체(이미지+description 직접 입력)는 정상 동작.
- 프로그램을 실행한 컴퓨터의 브라우저에서 `http://localhost:7860`에 접속합니다.
- 화면에서 이미지를 업로드하고 설명문을 입력해 감성/형용사 상위 5개를 확인합니다.
- 사용 중에는 실행 터미널을 열어 두며, 종료할 때는 터미널에서 `Ctrl+C`를 누릅니다.

## collect_description.py — description 수집용 경량 도구
모델 추론 없이(GPU 불필요) 이미지에 대한 description 텍스트만 받아서 `collected_descriptions.jsonl`에
저장하는 별도 도구. 새 데이터 라벨링/수집 시 사용.

```bash
python gradio/collect_description.py
```
