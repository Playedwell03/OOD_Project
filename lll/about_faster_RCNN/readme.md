Faster R-CNN Multi-Dataset Training Script
Faster R-CNN(ResNet50 + FPN)을 사용하여 여러 개의 COCO 포맷 데이터셋을 자동으로 반복 학습하는 스크립트입니다. 각 데이터셋마다 자동으로 모델을 학습시키고, best.pth, last.pth, log.jsonl을 저장합니다.

1. 사전 준비물 (Prerequisites)
 Python 버전
* Python 3.8 ~ 3.11 권장
 필수 패키지 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pycocotools
pip install opencv-python
pip install numpy
환경
* NVIDIA GPU (8GB VRAM 이상)
* Windows 또는 Linux
* CUDA 11.8 또는 12.x 권장

 2. 데이터셋 구조
각 데이터셋은 아래의 구조를 가져야 합니다.
dataset_name/
    train/
        images/
        train.json
    val/
        images/
        val.json
COCO JSON 형식 필수 포함 요소
* "images": [...]
* "annotations": [...]
* "categories": [...]
JSON "file_name"은 실제 이미지 파일과 정확히 일치해야 합니다.
예:
train/images/000123.jpg
train.json → "file_name": "000123.jpg"

 3. 클래스 개수 설정
코드 내 NUM_CLASSES는 다음 의미입니다:
NUM_CLASSES = 9   # 8 classes + background(0)
반드시 COCO JSON의 카테고리 개수와 일치해야 합니다.

 4. DATASETS 설정
코드 상단의 DATASETS 리스트만 수정하면 됩니다:
DATASETS = [
    ("A_dataset_aug_split/train", "A_dataset_aug_split/train.json",
     "A_dataset_aug_split/val",   "A_dataset_aug_split/val.json",   "A_dataset_aug_split"),

    ("dataset_a_aug_split/train", "dataset_a_aug_split/train.json",
     "dataset_a_aug_split/val",   "dataset_a_aug_split/val.json",   "dataset_a_aug_split"),

    ...
]
각 항목의 구성:
(train_root, train_json, val_root, val_json, run_name)

⚙ 5. 실행 방법
python train_frcnn_5_models.py

 6. 출력물 (Output Files)
각 데이터셋별로 아래 폴더가 생성됩니다:
outputs/<run_name>/
    best.pth       # 가장 낮은 val_loss 시점의 가중치
    last.pth       # 최근 epoch 저장본
    log.jsonl      # epoch, train_loss, val_loss 기록

 7. 모델 설명 (Faster R-CNN + ResNet50 FPN)
Faster R-CNN은 다음 구조로 이루어진 객체탐지 모델입니다.
Backbone: ResNet50
이미지에서 특징(feature)을 추출하는 부분.
 FPN (Feature Pyramid Network)
여러 해상도의 feature map을 결합하여 작은 객체(도로표지판 등)에 강함.
 RPN (Region Proposal Network)
이미지 안에서 "물체가 있을 법한 후보 영역"을 예측.
 ROI Head
각 후보 영역을
* 어떤 클래스인지
* 정확한 박스 위치 보정 을 수행함.

 8. 학습 시 자동 수행 기능
* AMP(Automatic Mixed Precision) 사용 → GPU 메모리 절약
* 매 epoch마다 train loss 계산
* quick_val_loss를 이용한 빠른 검증
* best model 자동 저장
* JSON 라인 로그 기록

 9. 실행 전 체크리스트
체크 항목	상태
Python 3.8~3.11 설치됨	☐
torch / torchvision 설치됨	☐
pycocotools 설치됨	☐
각 dataset의 구조가 올바른가	☐
JSON "file_name"이 이미지와 일치하는가	☐
NUM_CLASSES가 실제 클래스 개수와 맞는가	☐
GPU 학습 가능 여부 (torch.cuda.is_available()) 확인	☐
경로에 한글이 없는지 확인 (Windows 권장)	☐
 10. 참고: 주요 파일 설명
● train_frcnn_5_models.py
* 전체 학습 스크립트
* 여러 데이터셋을 자동으로 순회하며 학습
* DataLoader, COCO Reader, 모델 생성, 학습 루프까지 포함
● outputs/<run_name>/log.jsonl
각 줄이 다음 구조로 기록됩니다:
{"epoch": 1, "train_loss": 0.5532, "val_loss": 0.6173, "time": 1733648200.13}

 