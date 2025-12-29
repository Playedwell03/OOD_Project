import os
import json
import shutil
from tqdm import tqdm
import datetime

# ==========================================
# [설정] 경로를 확인해주세요
# ==========================================
SOURCE_DIR = 'coco_ood_cropped'         # 아까 생성한 폴더별 데이터 경로
FINAL_DIR = 'coco_ood_final'            # 최종적으로 저장될 경로 (한곳에 모인 버전)
OUTPUT_JSON = os.path.join(FINAL_DIR, 'ood_instances.json')

# 클래스 매핑 (아까와 동일)
CLASS_MAP = {
    'class0_cat': 1, 'class1_backpack': 2, 'class2_bumbrella': 3, 'class3_tie': 4,
    'class4_banana': 5, 'class5_apple': 6, 'class6_chair': 7, 'class7_couch': 8,
    'class8_bed': 9, 'class9_laptop': 10, 'class10_mouse': 11, 'class11_keyboard': 12,
    'class12_cell phone': 13, 'class13_microwave': 14, 'class14_oven': 15, 'class15_clock': 16,
}
# ==========================================

def merge_and_create_json():
    # 1. 최종 저장 폴더 생성
    images_dir = os.path.join(FINAL_DIR, 'images')
    if os.path.exists(FINAL_DIR):
        shutil.rmtree(FINAL_DIR)
    os.makedirs(images_dir)

    # 2. JSON 기본 구조
    coco_format = {
        "info": {"description": "COCO OOD Final Dataset", "year": 2025, "date_created": datetime.datetime.now().isoformat()},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 카테고리 등록
    for name, cid in CLASS_MAP.items():
        coco_format["categories"].append({"id": cid, "name": name, "supercategory": "ood_object"})

    print(f"🚀 데이터 병합 및 JSON 생성 시작...\n   소스: {SOURCE_DIR} -> 타겟: {images_dir}")

    img_id_cnt = 1
    ann_id_cnt = 1

    # 소스 폴더(클래스별 폴더) 순회
    for class_folder in os.listdir(SOURCE_DIR):
        folder_path = os.path.join(SOURCE_DIR, class_folder)
        
        if not os.path.isdir(folder_path) or class_folder not in CLASS_MAP:
            continue
            
        category_id = CLASS_MAP[class_folder]
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"👉 Moving '{class_folder}' ({len(image_files)} files)...")

        for img_file in tqdm(image_files):
            src_path = os.path.join(folder_path, img_file)
            
            # [중요] 파일명 중복 방지를 위해 '클래스명_파일명'으로 변경하여 저장
            # 예: cat_001.jpg
            new_file_name = f"{class_folder}_{img_file}"
            dst_path = os.path.join(images_dir, new_file_name)
            
            # 1. 파일 복사
            shutil.copy(src_path, dst_path)
            
            # 2. 이미지 정보 읽기 (OpenCV 없이 파일 사이즈만 체크하거나, 이전 코드처럼 읽기)
            # 여기선 정확성을 위해 cv2 로드 없이 os.stat 등으로 처리할 수도 있지만, 
            # BBox 생성을 위해 이미지 크기가 필요하므로 cv2나 PIL 사용 권장. 
            # (속도를 위해 여기서는 생략하고, 아까 생성된 이미지가 정사각형(IMG_SIZE)임을 가정한다면 고정값 써도 됨)
            # 하지만 안전하게 직접 읽겠습니다.
            import cv2
            img = cv2.imread(dst_path)
            if img is None: continue
            h, w = img.shape[:2]

            # 3. Images 정보 등록 (이제 파일명에 경로가 없음!)
            coco_format["images"].append({
                "id": img_id_cnt,
                "width": w,
                "height": h,
                "file_name": new_file_name  # 경로 없이 파일명만!
            })

            # 4. Annotations 등록
            coco_format["annotations"].append({
                "id": ann_id_cnt,
                "image_id": img_id_cnt,
                "category_id": category_id,
                "segmentation": [],
                "area": w * h,
                "bbox": [0, 0, w, h], # 이미지 전체가 박스
                "iscrowd": 0
            })

            img_id_cnt += 1
            ann_id_cnt += 1

    # JSON 저장
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(coco_format, f, indent=4)

    print(f"\n🎉 완료되었습니다!")
    print(f"📁 이미지 위치: {images_dir}")
    print(f"📝 라벨 파일: {OUTPUT_JSON}")

if __name__ == "__main__":
    merge_and_create_json()