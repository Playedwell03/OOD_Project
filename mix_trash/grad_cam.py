import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
# EigenCAM 임포트는 그대로 유지합니다.
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from scipy.stats import entropy as calc_entropy

# ------------------------
# 🔧 사용자 설정
# ------------------------
MODEL_PATH = "models/10per/weights/best.pt"
TEST_DIR = "A_tvt/test/images"
OUTPUT_DIR = "gradcam_results/10per_layer_-4" # 결과 폴더 이름 변경
CSV_PATH = os.path.join(OUTPUT_DIR, "gradcam_stats.csv")

# ------------------------
# 🚀 기본 준비
# ------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class YOLOV5ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        result = self.model(x)
        # 튜플 출력을 가로채서 첫 번째 텐서만 반환합니다.
        if isinstance(result, tuple):
            return result[0]
        return result

model = YOLO(MODEL_PATH)
yolo_model = model.model.to(device).eval()

wrapped_model = YOLOV5ModelWrapper(yolo_model)

# ✨ CHANGED: 분석할 대상 레이어를 -2에서 -4로 변경합니다.
# ------------------------
target_layer = wrapped_model.model.model[-4]
# ------------------------
cam = EigenCAM(model=wrapped_model, target_layers=[target_layer])

stats = []

# ------------------------
# 테스트 이미지 순회
# ------------------------
for file_name in tqdm(os.listdir(TEST_DIR)):
    if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(TEST_DIR, file_name)
    rgb_img = cv2.imread(img_path)[..., ::-1]
    rgb_img = cv2.resize(rgb_img, (640, 640))
    rgb_img_float = rgb_img.astype(np.float32) / 255.0
    input_tensor = preprocess_image(rgb_img_float, mean=[0,0,0], std=[1,1,1]).to(device)

    # (통계 저장을 위한 예측 정보)
    results = model.predict(img_path, verbose=False)
    pred_class_for_stats = -1
    if len(results[0].boxes) > 0:
        top_idx = results[0].boxes.conf.argmax()
        pred_class_for_stats = int(results[0].boxes.cls[top_idx].item())

    # 2️⃣ EigenCAM 계산 (targets 인자 없이 호출)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]

    # 3️⃣ 시각화 및 저장
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    cv2.imwrite(save_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

    # 4️⃣ 정량 지표 계산
    cam_sum = np.sum(grayscale_cam)
    if cam_sum == 0:
        continue
        
    p = grayscale_cam.flatten() / cam_sum
    ent = calc_entropy(p)
    var = np.var(grayscale_cam)

    stats.append({
        'file': file_name,
        'entropy': ent,
        'variance': var,
        'pred_class': pred_class_for_stats
    })

# ------------------------
# 📊 CSV 저장
# ------------------------
if stats:
    df = pd.DataFrame(stats)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n✅ Eigen-CAM 결과 저장 완료: {OUTPUT_DIR}")
    print(f"📈 통계 CSV 저장 완료: {CSV_PATH}")
else:
    print("\n⚠️ No stats were generated.")