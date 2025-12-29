import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import shutil

# ==========================================
# [사용자 설정 구간] 경로를 수정하세요
# ==========================================
# 1. COCO 데이터셋 경로 (이미지가 있는 폴더)
DATA_DIR = './coco/train2017' 
# 2. COCO 어노테이션 파일 경로 (.json)
ANN_FILE = './coco/annotations/instances_train2017.json'
# 3. 크롭된 이미지가 저장될 경로
SAVE_DIR = './coco_ood_cropped'

# 4. 가져올 COCO 클래스와 매핑할 나의 OOD 클래스 이름
# 형식: {'COCO클래스이름': '저장될_폴더이름'}
TARGET_CLASSES = {
    'cat': 'class0_cat',       # 사용자 Class 0
    'backpack': 'class1_backpack',  # 사용자 Class 1 (가방 대신 여행가방 예시)
    'umbrella': 'class2_bumbrella',        # 사용자 Class 2
    'tie': 'class3_tie',
    'banana': 'class4_banana',
    'apple': 'class5_apple',
    'chair': 'class6_chair',
    'couch': 'class7_couch',
    'bed': 'class8_bed',
    'laptop': 'class9_laptop',
    'mouse': 'class10_mouse',
    'keyboard': 'class11_keyboard',
    'cell phone': 'class12_cell phone',
    'microwave': 'class13_microwave',
    'oven': 'class14_oven',
    'clock': 'class15_clock',
}

# 5. 생성할 이미지 크기 (모델 입력 크기에 맞추거나, 넉넉하게 640 등)
IMG_SIZE = 416
# ==========================================

def make_square_with_padding(image, target_size, pad_value=(0, 0, 0)):
    """
    이미지 비율을 유지하면서 검은색(pad_value) 배경을 추가해 정사각형으로 만듭니다.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 리사이즈
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 검은색 캔버스 생성
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)
    
    # 중앙 배치 좌표 계산
    x_center = (target_size - new_w) // 2
    y_center = (target_size - new_h) // 2
    
    # 이미지 붙여넣기
    canvas[y_center:y_center+new_h, x_center:x_center+new_w] = resized_image
    return canvas

def main():
    # 저장 디렉토리 초기화
    if os.path.exists(SAVE_DIR):
        shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)

    # COCO API 로드
    coco = COCO(ANN_FILE)

    print(f"\n🚀 COCO OOD Cropping 시작... (Target: {list(TARGET_CLASSES.keys())})")

    for coco_cls, save_name in TARGET_CLASSES.items():
        # 해당 클래스의 폴더 생성
        class_dir = os.path.join(SAVE_DIR, save_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # COCO에서 해당 카테고리 ID 가져오기
        catIds = coco.getCatIds(catNms=[coco_cls])
        if not catIds:
            print(f"⚠️ 경고: '{coco_cls}' 클래스를 COCO에서 찾을 수 없습니다. 스킵합니다.")
            continue
            
        # 해당 카테고리가 포함된 이미지 ID 가져오기
        imgIds = coco.getImgIds(catIds=catIds)
        
        print(f"👉 Processing '{coco_cls}' -> '{save_name}' (Found {len(imgIds)} images)")
        
        count = 0
        for imgId in tqdm(imgIds):
            img_info = coco.loadImgs(imgId)[0]
            img_path = os.path.join(DATA_DIR, img_info['file_name'])
            
            # 이미지 로드
            img = cv2.imread(img_path)
            if img is None:
                continue # 이미지 파일이 없으면 패스
                
            # 해당 이미지 안의 어노테이션(BBox) 가져오기
            annIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            for i, ann in enumerate(anns):
                # BBox 좌표 (x, y, w, h)
                x, y, w, h = map(int, ann['bbox'])
                
                # 예외 처리: 박스가 너무 작거나 이미지 벗어난 경우
                if w < 10 or h < 10: continue
                
                # 이미지 Crop (좌표 보정)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(img.shape[1], x + w)
                y2 = min(img.shape[0], y + h)
                
                cropped_obj = img[y1:y2, x1:x2]
                
                if cropped_obj.size == 0: continue

                # 검은 배경 중앙에 배치 (Letterbox 방식)
                final_img = make_square_with_padding(cropped_obj, IMG_SIZE)
                
                # 파일 저장 (파일명: 원본이름_객체번호.jpg)
                file_name = f"{os.path.splitext(img_info['file_name'])[0]}_{count}.jpg"
                save_path = os.path.join(class_dir, file_name)
                cv2.imwrite(save_path, final_img)
                count += 1
                
        print(f"✅ '{save_name}' 완료: 총 {count}장 저장됨.\n")

    print(f"🎉 모든 작업이 완료되었습니다! 저장 경로: {SAVE_DIR}")

if __name__ == "__main__":
    main()