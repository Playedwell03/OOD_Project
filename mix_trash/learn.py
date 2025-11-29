from ultralytics import YOLO

# YOLOv5s 모델 구조만 로드 (가중치는 랜덤 초기화)
model = YOLO('yolov5s.yaml')
# yolov5s.pt, yolov5s.yaml

# 학습
model.train(
    data='added_fish_A6/50per/data.yaml',
    epochs=50,
    imgsz=416,
    batch=16,
    device='cpu',
    name='50per',
    project='models',
    patience=0,
    optimizer='SGD',
    workers=4
)