# 로컬 DINO 모델 패키지

모델 등록은 신뢰하는 로컬 가중치만 대상으로 합니다. remote-code나 임의 pickle 실행은 허용하지 않습니다. 기존 .pt는 torch.load(weights_only=True)로만 읽으며 실패 시 안전하지 않은 로딩으로 전환하지 않습니다. 화면에서 폴더를 선택하면 파일 구성과 해시를 확인하고, 복사 후 네트워크 없이 실제 로딩·합성 이미지 검사를 통과한 경우에만 등록합니다.

## 기존 DINO A·B + RBF 모델 (추론 및 분류 헤드 추가 학습)

```text
선택할 폴더/
├── backbone/                       # dinov3l이라는 폴더명도 인식
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors           # 또는 index.json + 모든 safetensors 분할
└── weights/                        # runtime_weights 또는 최상위 배치도 인식
    ├── dinov3_end_to_end_strict.pt
    ├── dinov3_pseudopretrain_strict.pt
    └── image_residual.pt
```

이 구조는 package.json을 직접 작성하지 않아도 자동 감지합니다. 동일 가중치가 최상위와 weights에 중복되면 사용할 파일 하나만 두세요. RBF 하나 또는 A/B 둘만으로는 등록할 수 없습니다. 별도의 원본 백본이 RBF 특징 추출에 필요합니다. 백본은 원본 safetensors와 설정을 사용하며 임의 대체하지 않습니다.

추론은 원본 코드의 0.46 × A + 0.54 × B + RBF 잔차입니다. A/B는 Resize(256), CenterCrop(224), ImageNet 정규화와 원본·좌우·상하·양방향 반전 4개 뷰의 평균 로짓을 sigmoid 처리합니다. RBF에는 원본 백본의 저장된 이미지 전처리와 CLS+평균 patch 특징을 사용합니다. GT·원설명문·파일 ID로 점수를 보정하지 않습니다.

FULL_MODEL에는 원본 전체 파일이 포함되어 자동 등록 상태로 제공됩니다. 로딩 시험 및 소량 추가 학습 검증은 전체 평가셋 성능 검증과 다릅니다. 원본 배포 기록 자체도 평가 표본 학습 재사용을 포함합니다. 0.7부터 기존 A/B 헤드를 이어 학습할 수 있습니다. DINO 백본·RBF·앙상블 비율은 고정하며 원본 연구 전체 학습 재현과는 구분합니다. 기존 .pt에 든 Linux 경로 메타데이터는 PurePosixPath로만 읽으며 임의 pickle 실행으로 전환하지 않습니다.

추가 학습 모델의 `manifest.json`은 `engine=legacy_dino_rbf_v1`, `base_model=<공통 원본 ID>`, `parent_model=<직전 학습 출발 모델 ID>`를 기록합니다. `head.safetensors`에는 `a.*`, `b.*` 키로 두 헤드를 저장합니다. 원본 모델과 새 헤드를 함께 불러와 실행합니다. 이전 0.4~0.6 원본 manifest의 training_supported=false는 당시 기능 제한 기록이며 0.7은 원본을 덮어쓰지 않고 실행 코드에서 지원 여부를 판단합니다.

## 단일 DINO 추가 학습 모델

```text
workspace/models/dino_baseline/
├── package.json
├── backbone/
│   ├── config.json                  # model_type: dinov3_vit
│   ├── preprocessor_config.json
│   └── model.safetensors            # 분할 safetensors도 가능
└── head.safetensors                 # 학습된 헤드가 있을 경우
```

`config.json`과 전처리 설정은 해당 backbone의 실제 저장본이어야 합니다. 원래와 다른 전처리를 임의로 섞으면 안 됩니다. 모든 파일은 로컬에 있어야 하며 링크를 통한 외부 참조는 금지합니다.

```json
{
  "engine": "dino_head_v1",
  "name": "DINO 이미지 모델",
  "has_head": false,
  "labels": ["강한","고급스러운","고전적인","귀여운","깔끔한","단순한","독특한","부드러운","섬세한","소박한","신비한","우아한","원초적인","유려한","익살스러운","자연적인","장식적인","조화로운","풍성한","현대적인","화려한","활동적인"],
  "provenance": "기반 모델만 등록. 이전 학습 노출은 별도 확인 필요",
  "trained_pixel_hashes": []
}
```

`has_head:false`이면 학습은 가능하지만 추론·운영 모델 적용은 불가능합니다. 사용자 학습 전까지 임의 점수를 감성 예측으로 내보내지 않습니다.

헤드는 FP32 `Sequential(LayerNorm(2H), Linear(2H,1024), GELU(), Dropout(0.2), Linear(1024,22))`입니다. safetensors 키는 `0.weight`, `0.bias`, `1.weight`, `1.bias`, `4.weight`, `4.bias`입니다. 기존 `.pt`의 모양과 라벨 순서가 같아 보인다는 이유만으로 연결해서는 안 됩니다. 실제 전처리·토큰 추출·점수 결합·예측 재현까지 검증해야 합니다.

새 학습 후보는 backbone을 중복 복사하지 않고 동일 models 안의 기반 패키지를 ID로 참조합니다. 따라서 윈도우 PC를 옮길 때에는 새 헤드만이 아니라 `workspace/models` 전체를 함께 옮겨 주세요. 기본 모델 적용 변경은 가중치 파일 자체를 수정하지 않습니다.
# 추가 형식: bert5_fusion_v1 (0.11)

이미지+설명문을 함께 사용하는 5종 융합 모델입니다. `backbone/`에는 로컬 DINOv3, `weights/`에는 klue.pt·xlmr.pt·kcbert.pt·mbert.pt·kobigbird.pt, `text/<tag>/`에는 각각의 config/tokenizer 파일, 루트에는 `fusion_config.json`이 필요합니다. `package.json`의 engine은 bert5_fusion_v1, labels는 동일 22개 순서, has_head는 true로 지정합니다. 원본 .pt는 각 텍스트 백본과 융합부의 state_dict이며 weights_only=True로 읽습니다. 폴더 등록 시 파일과 실제 로딩을 검사합니다.

추가 학습 모델은 base_model로 원본을 참조하고, head.safetensors에 5종 융합부만 저장합니다. parent_model은 학습 계보입니다. 설명문은 명시적으로 입력하며 GT나 등록 이미지 ID로 자동 보충하지 않습니다. 원본 80% 기록은 이미지+원설명문 조건이며 현재 연결 모델의 전체 재평가 결과가 아닙니다.

# 0.14 프로그램용 가중치 백업 ZIP

가중치 보관함의 **다른 이름으로 저장**은 `pattern-model-backup-v1` ZIP을 만듭니다. `bundle.json`에 선택 모델, 실행에 필요한 모델 참조, 파일별 SHA-256·크기를 기록하고 각 모델 ID 하위에 필요한 파일을 넣습니다. 원본 백본을 참조하는 추가학습 모델도 해당 원본을 함께 보존합니다. 추론 입력 이미지나 정답 CSV는 포함하지 않습니다.

이 ZIP은 **저장한 가중치 ZIP 불러오기**로 등록하며 직접 풀어서 폴더 등록할 필요가 없습니다. 새 모델 ID를 부여하고 원본 참조를 다시 연결하며, 임시 보관함에서 실제 로딩 검사를 마친 뒤 등록합니다. 현재 적용 모델은 변경하지 않습니다. 기존 폴더 등록 형식과 아래 규칙은 그대로 유지됩니다.

