from ultralytics import YOLO
import os

# 모델 로드
model = YOLO('models/50per_batch8/weights/best.pt')  
model.to('cpu')

#####################
# 1. 성능 평가 수행 #
#####################
metrics = model.val(
    data='50per/data.yaml',
    split='test',
    imgsz=416,
    batch=8,
    project='result_with_coco',
    name='50per_batch8',
    exist_ok=True,
    # conf = 0.05
    # iou = 0.7
)

# 평가 지표 출력
print("\n📊 [모델 평가 지표]")
print(f"mAP@0.5:          {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:     {metrics.box.map:.4f}")
print(f"Precision (mean): {metrics.box.mp:.4f}")
print(f"Recall (mean):    {metrics.box.mr:.4f}")

# 평가 지표 파일 저장
output_dir = os.path.join('result_with_coco', '50per_batch8')
os.makedirs(output_dir, exist_ok=True)
metrics_path = os.path.join(output_dir, 'metrics.txt')

with open(metrics_path, 'w') as f:
    f.write("[모델 평가 지표]\n")
    f.write(f"mAP@0.5:          {metrics.box.map50:.4f}\n")
    f.write(f"mAP@0.5:0.95:     {metrics.box.map:.4f}\n")
    f.write(f"Precision (mean): {metrics.box.mp:.4f}\n")
    f.write(f"Recall (mean):    {metrics.box.mr:.4f}\n")

#######################################
# 2. 예측 결과 이미지 저장 (시각화용) #
#######################################
results = model.predict(
    source='50per/test/images',
    imgsz=416,
    save=True,
    project='predict_imgs_coco',
    name='50per_batch8',
    exist_ok=True,
    batch=8,
    device='cpu',
    # iou = 0.7
    # conf=0.05
)

print("\n✅ 예측 이미지 저장 완료!")
print("✅ 평가 지표 파일 저장 완료!")
 