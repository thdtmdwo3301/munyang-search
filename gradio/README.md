# Gradio 데모

## app_gradio.py — 감성분류 데모 (모델 추론)
이미지 + description을 넣으면 top-5 감성 라벨/점수를 보여준다. `../inference/`의 config.json과
weights/를 그대로 불러와 쓰므로(가중치 중복 보관 안 함) 실행 전 `../inference/`가 있어야 한다.

```bash
pip install -r ../inference/requirements.txt gradio
python app_gradio.py
```
- 데이터셋에 있는 이미지(파일명이 원본 그대로인 경우)를 올리면 description을 자동으로
  채워주는 기능이 있는데, 이건 `../train/dataset.py`가 참조하는 원본 ETRI 데이터셋이 로컬에
  있어야 동작함. 없어도 모델 추론 자체(이미지+description 직접 입력)는 정상 동작.
- 기본으로 `0.0.0.0:7860`에 뜨고 `share=True`라 공개 gradio.live 링크도 같이 생성됨(최대 7일
  유효). 공개 링크가 필요 없으면 `app_gradio.py` 맨 아래 `demo.launch(...)`에서 `share=False`로
  바꿀 것.

## collect_description.py — description 수집용 경량 도구
모델 추론 없이(GPU 불필요) 이미지에 대한 description 텍스트만 받아서 `collected_descriptions.jsonl`에
저장하는 별도 도구. 새 데이터 라벨링/수집 시 사용.

```bash
python collect_description.py
```
