import os
import shutil
import json
import math
import random
from pycocotools.coco import COCO
from tqdm import tqdm

# ==========================================
# [사용자 설정] 경로를 프로젝트에 맞게 수정하세요
# ==========================================

# 1. 원본 데이터셋 경로 (건드리지 않고 읽기만 함)
ORIGINAL_DATASET_DIR = 'A_tvt'

# 2. 새로 생성될 데이터셋 경로 (여기에 복사 후 OOD가 추가됨)
# 에러 로그에 있던 폴더명('ood_10per_cropped')을 그대로 사용했습니다.
NEW_OUTPUT_DIR = 'ood_50per_cropped/train' 

# 3. 가져올 OOD 소스 (coco_ood_final)
COCO_IMG_DIR = 'coco_ood_final/images'
COCO_ANN_FILE = 'coco_ood_final/ood_instances.json'

# 4. 추가할 OOD 비율 (예: 0.1 = 10%)
TARGET_OOD_RATIO = 0.5

# 5. 클래스 매핑
# { A_tvt의 ID : 'coco_ood_final의 카테고리 이름' }
CATEGORY_MAPPING = {
    0: 'class0_cat',       
    1: 'class1_backpack',
    2: 'class2_bumbrella',
    3: 'class3_tie',
    4: 'class4_banana',
    5: 'class5_apple',
    6: 'class6_chair',
    7: 'class7_couch',
    8: 'class8_bed',
    9: 'class9_laptop',
    10: 'class10_mouse',
    11: 'class11_keyboard',
    12: 'class12_cell phone',
    13: 'class13_microwave',
    14: 'class14_oven',
    15: 'class15_clock'
}
# ==========================================

def main():
    # ---------------------------------------------------------
    # 0. 작업 폴더 초기화 및 원본 복사
    # ---------------------------------------------------------
    if os.path.exists(NEW_OUTPUT_DIR):
        print(f"⚠️ 기존 출력 폴더({NEW_OUTPUT_DIR})가 존재하여 삭제 후 다시 생성합니다...")
        shutil.rmtree(NEW_OUTPUT_DIR)
    
    # 상위 폴더(ood_10per_cropped)가 없으면 생성
    os.makedirs(os.path.dirname(NEW_OUTPUT_DIR), exist_ok=True)

    print(f"📦 원본 데이터 복사 중... ({ORIGINAL_DATASET_DIR} -> {NEW_OUTPUT_DIR})")
    shutil.copytree(ORIGINAL_DATASET_DIR, NEW_OUTPUT_DIR)
    print("✅ 복사 완료! 이제 사본 데이터셋에 작업을 시작합니다.")

    # 작업 경로는 이제 '새로운 폴더'가 됩니다.
    BASE_JSON_PATH = os.path.join(NEW_OUTPUT_DIR, 'train.json')
    BASE_IMG_DIR = os.path.join(NEW_OUTPUT_DIR, 'images')

    # 1. 복사된 JSON 로드
    with open(BASE_JSON_PATH, 'r') as f:
        base_data = json.load(f)

    # 2. 기존 데이터 분석 (ID 충돌 방지용)
    # 이미지가 하나도 없을 경우를 대비해 default=0 처리
    max_img_id = max([img['id'] for img in base_data['images']], default=0)
    max_ann_id = max([ann['id'] for ann in base_data['annotations']], default=0)
    
    print(f"📊 기존 데이터 분석 완료 (Max Image ID: {max_img_id})")

    # 클래스별 기존 객체 수 카운트
    orig_counts = {}
    for cid in CATEGORY_MAPPING.keys():
        count = sum(1 for ann in base_data['annotations'] if ann['category_id'] == cid)
        orig_counts[cid] = count

    # 3. COCO API 로드
    coco = COCO(COCO_ANN_FILE)
    
    print("\n🚀 병합 시작 (OOD 데이터 주입)...")
    
    added_img_count = 0
    added_ann_count = 0

    for target_cls_id, coco_cat_name in CATEGORY_MAPPING.items():
        n_orig = orig_counts.get(target_cls_id, 0)
        
        # 목표 개수 계산
        if n_orig == 0:
            target_ood_count = 0
        else:
            target_ood_count = math.ceil(n_orig * (TARGET_OOD_RATIO / (1 - TARGET_OOD_RATIO)))
        
        if target_ood_count == 0:
            continue
            
        print(f"👉 Class {target_cls_id} ({coco_cat_name}): 기존 {n_orig}개 -> OOD {target_ood_count}개 추가 예정")

        catIds = coco.getCatIds(catNms=[coco_cat_name])
        imgIds = coco.getImgIds(catIds=catIds)
        random.shuffle(imgIds)

        current_added_inst = 0
        pbar = tqdm(total=target_ood_count, desc=f"   Merging {coco_cat_name}")
        
        for img_id in imgIds:
            if current_added_inst >= target_ood_count:
                break
            
            img_info = coco.loadImgs(img_id)[0]
            annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            if not anns: continue

            # --- [파일 복사 및 폴더 생성] ---
            new_file_name = f"ood_{img_info['file_name']}" 
            src_path = os.path.join(COCO_IMG_DIR, img_info['file_name'])
            dst_path = os.path.join(BASE_IMG_DIR, new_file_name)
            
            # ✨ [핵심 수정] 저장할 폴더(images)가 없으면 강제로 생성!
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # 파일 복사 (없을 때만)
            if not os.path.exists(dst_path):
                shutil.copy(src_path, dst_path)
            
            # --- [JSON 데이터 추가: Image] ---
            max_img_id += 1
            new_img_entry = {
                "id": max_img_id,
                "file_name": new_file_name,
                "width": img_info['width'],
                "height": img_info['height'],
                "date_captured": img_info.get('date_captured', ""),
                "license": img_info.get('license', 0)
            }
            base_data['images'].append(new_img_entry)
            added_img_count += 1

            # --- [JSON 데이터 추가: Annotation] ---
            for ann in anns:
                max_ann_id += 1
                new_ann_entry = {
                    "id": max_ann_id,
                    "image_id": max_img_id,
                    "category_id": target_cls_id,
                    "bbox": ann['bbox'],
                    "area": ann['area'],
                    "segmentation": ann['segmentation'],
                    "iscrowd": ann['iscrowd']
                }
                base_data['annotations'].append(new_ann_entry)
                current_added_inst += 1
                pbar.update(1)
        
        pbar.close()
        added_ann_count += current_added_inst

    # 4. 최종 JSON 저장 (새로운 폴더의 json 파일 덮어쓰기)
    with open(BASE_JSON_PATH, 'w') as f:
        json.dump(base_data, f, indent=4)

    print("\n🎉 모든 작업 완료!")
    print(f"📁 결과 저장 경로: {NEW_OUTPUT_DIR}")
    print(f"📝 라벨 파일: {BASE_JSON_PATH}")

if __name__ == '__main__':
    main()