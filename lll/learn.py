from ultralytics import YOLO

# YOLOv5s 모델 구조만 로드 (가중치는 랜덤 초기화)
model = YOLO('yolov5s.yaml')

# 학습 (GPU 사용)
model.train(
    data='30per/data.yaml',
    epochs=50,
    imgsz=416,
    batch=4,
    device='0',          # ✅ GPU 0번 사용 CPU는 1
    name='30per_batch4',
    project='models',
    patience=0,
    optimizer='SGD',
    workers=8
)
