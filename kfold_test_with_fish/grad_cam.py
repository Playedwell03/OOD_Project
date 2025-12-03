import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import glob
import traceback # 👈 [추가됨] 상세한 오류 추적용

# --- [수정됨] 1. YOLOv8/v5용 '튜플' 해결 래퍼 ---
# 이 클래스가 "AttributeError: 'tuple' object" 오류를 해결합니다.
class YOLOv8CAMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # model.model (실제 PyTorch nn.Module)
    
    def forward(self, x):
        # 모델을 실행합니다.
        # YOLOv8/v5의 model.model(x)는 튜플을 반환합니다.
        # 예: ( (det_head_1, det_head_2, ...), (seg_head_1, ...) )
        # 또는 ( (det_head_1, ...), other_internal_tensors )
        model_output = self.model(x)
        
        # grad-cam은 텐서가 필요한데, 튜플의 첫 번째 요소가
        # (텐서 리스트)일 수 있습니다.
        if isinstance(model_output, tuple):
            # 튜플의 첫 번째 요소 (det_head_outputs)를 가져옵니다.
            # 이 요소가 리스트나 튜플일 수 있습니다.
            main_output = model_output[0]
            
            # 만약 main_output이 리스트/튜플이면, 
            # (아마도 3개의 감지 헤드)
            # 이 텐서들을 하나로 합칩니다(concatenate).
            if isinstance(main_output, (list, tuple)):
                # (Batch, Heads, Features...) -> (Batch, All_Features)
                # 여기서 감지 헤드의 특징(feature)을 합칩니다.
                # 예: 3개의 (Batch, 80, 80, 255) 텐서를 합칩니다.
                # 간단히 마지막 텐서(가장 큰 특징 맵)만 반환하거나,
                # 모두 합쳐서 반환합니다. 여기서는 마지막 것을 시도합니다.
                return main_output[-1] # 👈 튜플/리스트의 마지막 텐서 반환
            else:
                return main_output # 👈 튜플의 첫 번째 요소가 텐서인 경우
        else:
            return model_output # 이미 텐서인 경우

# --- (Unchanged) 이미지 전처리 함수 ---
def load_and_preprocess_pil(image_path, target_size=416):
    try:
        pil_img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"    ⚠️ PIL 이미지 로드 실패: {image_path}, Error: {e}")
        return None, None
    preprocess = T.Compose([
        T.Resize((target_size, target_size)),
        T.ToTensor()
    ])
    tensor = preprocess(pil_img).unsqueeze(0)
    rgb_img_norm = np.array(pil_img.resize((target_size, target_size))) / 255.0
    return rgb_img_norm, tensor

# --- 2. ⚠️ 사용자 설정 ---
MODEL_PATHS = {
    # "A_0per": r"models_k10_runs_A/run_1_test/weights/best.pt",
    "A_30per": r"models_k10_runs_30per/run_1_test/weights/best.pt"
}
IMAGE_FOLDERS = {
    "normal_test_images": r"A_k10_runs/run_3_test/test/images",
    "ood_images": r"fishes_for_gradcam" 
}
IMAGE_EXTENSIONS = ('*.jpg', '*.png', '*.jpeg')
OUTPUT_DIR = "grad_cam_comparison_all"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 416
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"--- Grad-CAM 비교 시작 (Device: {DEVICE}) ---")

# --- 3. 모델 루프 ---
for model_name, model_path in MODEL_PATHS.items():
    print(f"\nProcessing Model: {model_name}")
    
    if not os.path.exists(model_path):
        print(f"  ⚠️ 모델을 찾을 수 없음: {model_path}. 건너뜁니다.")
        continue
        
    model_orig = YOLO(model_path)
    model_orig.to(DEVICE)
    model_orig.eval()

    # 3.2. [NEW] 모델을 'Wrapper'로 감싸기
    wrapped_model = YOLOv8CAMWrapper(model_orig.model)
    
    # 3.3. [수정됨] Grad-CAM 타겟 레이어 설정
    try:
        # YOLOv5 백본의 마지막 레이어(SPPF)를 타겟으로 합니다.
        target_layer = [model_orig.model.model[9]] # 👈 [8]에서 [9] (SPPF)로 변경
    except Exception as e:
        print(f"  ❌ 타겟 레이어 설정 실패: {e}. 모델 구조가 다를 수 있습니다.")
        continue
        
    # 3.4. [NEW] EigenCAM 인스턴스 생성
    cam = EigenCAM(
        model=wrapped_model,       # 👈 '튜플' 오류를 해결한 래퍼 모델
        target_layers=target_layer
    )

    # --- 4. 폴더 루프 ---
    for folder_type, folder_path in IMAGE_FOLDERS.items():
        print(f"  Processing folder: '{folder_type}' ({folder_path})")
        final_save_dir = os.path.join(OUTPUT_DIR, model_name, folder_type)
        os.makedirs(final_save_dir, exist_ok=True)

        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        if not image_files:
            print(f"    ⚠️ '{folder_path}'에서 이미지를 찾을 수 없습니다. 건너뜁니다.")
            continue
            
        print(f"    -> {len(image_files)}개의 이미지를 찾았습니다. CAM 생성을 시작합니다...")

        # --- 5. 파일 루프 ---
        for i, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            
            if (i+1) % 100 == 0:
                 print(f"    [{i+1}/{len(image_files)}] Processing {filename}...")
            
            try:
                rgb_img_norm, input_tensor = load_and_preprocess_pil(img_path, IMG_SIZE)
                if rgb_img_norm is None: continue
                input_tensor = input_tensor.to(DEVICE)

                grayscale_cam = cam(input_tensor=input_tensor)[0, :]
                
                cam_image = show_cam_on_image(
                    rgb_img_norm,
                    grayscale_cam,
                    use_rgb=True
                )
                
                final_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
                output_filename = f"cam_{filename}"
                save_path = os.path.join(final_save_dir, output_filename)
                cv2.imwrite(save_path, final_image_bgr)

            except Exception as e:
                # 
                # ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
                # 
                #           ✅✅✅ 이 부분이 수정되었습니다! ✅✅✅
                #           오류가 발생하면 '상세한' 오류 내용을 출력합니다.
                # 
                # ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
                # 
                print(f"    -> ❌ {filename} 처리 중 오류 발생:")
                traceback.print_exc() # 👈 상세한 오류 스택 출력

        print(f"    ✅ '{folder_type}' 폴더 처리 완료.")

print("\n🎉 모든 CAM 비교 작업 완료!")