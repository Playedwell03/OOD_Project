

# 📘 README: Faster R-CNN 모델들의 B 데이터셋 COCO 평가 스크립트

이 스크립트는 **Faster R-CNN(ResNet50-FPN)** 모델들을 B 데이터셋 전체(`B_all.json`) 기준으로 **COCO mAP 평가**하는 도구입니다.

> 👉 **YOLO 부분은 요청에 따라 제거(비활성화)하고, 관련 구간에 명확한 주석을 추가해둠.**
> 필요한 경우 다시 활성화해서 비교 평가를 진행할 수 있습니다.

---

# 🚀 1. 사전 준비물 (Prerequisites)

## ✔ Python 버전

* Python 3.8 ~ 3.11 권장

## ✔ 필수 패키지 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pycocotools
pip install opencv-python
pip install numpy
```

추가적으로 본 스크립트는 **COCOeval**을 통해 평가하기 때문에
`pycocotools` 설치가 필수입니다.

## ✔ 권장 환경

* NVIDIA GPU (8GB VRAM 권장)
* Windows 또는 Linux
* CUDA 11.8 / 12.x 환경

---

# 📁 2. 데이터셋 준비

### 평가 대상 데이터셋 = **B_dataset**

필수 파일 구조:

```
B_dataset/
    images/
        0001.jpg
        0002.jpg
        ...
    B_all.json
```

`B_all.json`은 **COCO format**이어야 하며 필수 필드는 다음과 같습니다:

* `"images"`
* `"annotations"`
* `"categories"`

### ⚠ 주의

`file_name`은 실제 이미지 파일명과 반드시 일치해야 합니다.

---

# 🧠 3. Faster R-CNN 체크포인트 준비

코드 상단에서 FRCNN_CKPTS 딕셔너리만 수정하면 됩니다:

```python
FRCNN_CKPTS = {
    "A_dataset_aug_split":   r".../outputs/A_dataset_aug_split/best.pth",
    "dataset_a_aug_split":   r".../outputs/dataset_a_aug_split/best.pth",
    ...
}
```

각 항목의 key는 “모델 이름(=결과 표기용)”이고,
value가 실제 **best.pth** 파일 경로입니다.

---

# 🔢 4. Faster R-CNN 클래스 수 설정

학습 시 **NUM_CLASSES_FRCNN = 9** 으로 사용했다면 아래도 동일해야 합니다:

```python
NUM_CLASSES_FRCNN = 9
```

즉,

```
8개 실제 클래스 + 1개 background = 9
```

---

# ⚙ 5. 스크립트 실행 방법

```bash
python eval_frcnn_B_dataset.py
```

실행하면 다음 순서가 진행됩니다:

1. Faster R-CNN 모델들 로드
2. B_dataset 전체에 대해 inference
3. COCOeval 계산
4. 각 모델별 결과 저장
5. summary JSON 생성
6. 화면에 표 형태로 출력

---

# 📤 6. 출력물 (Output Files)

```
eval_B_all/
    frcnn/
        <run_name>_pred_B.json          ← COCO detection format 결과
        <run_name>_pred_B_per_class.csv ← 클래스별 AP
    summary_B_all.json                 ← Faster R-CNN 전체 모델 성능 요약
```

---

# 📊 7. 출력되는 지표

각 모델별로 다음 mAP가 출력됩니다:

| metric         | 의미                         |
| -------------- | -------------------------- |
| `AP@[.50:.95]` | COCO 공식 메인 mAP             |
| `AP@0.50`      | IoU=0.50 기준 Pascal VOC mAP |
| `AP@0.75`      | 더 엄격한 IoU=0.75 기준 mAP      |

---
