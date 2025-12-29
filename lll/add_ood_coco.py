import os
import shutil
import glob
from pycocotools.coco import COCO
from tqdm import tqdm
import math

# --- ⚠️ 사용자 설정 ---

# 1. 원본 라벨 경로 (개수 파악용)
ORIGINAL_LABEL_DIR = 'A_tvt/train/labels'

# 2. COCO 데이터셋 경로 (train2017 권장)
COCO_IMG_DIR = 'coco_ood_final/train2017'
COCO_ANN_FILE = 'coco/annotations/instances_train2017.json'

# 3. 저장할 경로
OUTPUT_IMG_DIR = 'coco_ood_10/images'
OUTPUT_LABEL_DIR = 'coco_ood_10/labels'

# 4. 목표 OOD 비율 (예: 0.7 = 전체 객체 수의 70%)
TARGET_OOD_RATIO = 0.1

# 5. 매핑 (Class ID : COCO 카테고리)
CATEGORY_MAPPING = {
    0: 'cat',
    1: 'backpack',
    2: 'umbrella',
    3: 'tie',
    4: 'banana',
    5: 'apple',
    6: 'chair',
    7: 'couch',
    8: 'bed',
    9: 'laptop',
    10: 'mouse',
    11: 'keyboard',
    12: 'cell phone',
    13: 'microwave',
    14: 'oven',
    15: 'clock'
}
# ---------------------

def count_original_instances(label_dir):
    """원본 라벨 파일 내의 '줄(Line) 수'를 세서 객체 개수를 파악합니다."""
    class_counts = {}
    txt_files = glob.glob(os.path.join(label_dir, '*.txt'))
    print(f"📂 원본 데이터 객체 수 세는 중... ({len(txt_files)} 파일)")
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
    return class_counts

def convert_coco_bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return [x_center, y_center, w_norm, h_norm]

def main():
    # 1. 원본 객체 개수 파악
    orig_counts = count_original_instances(ORIGINAL_LABEL_DIR)
    
    # 2. 목표 객체 개수 계산
    needed_ood_counts = {}
    print("-" * 50)
    print(f"📊 목표: 객체(Instance) 기준 {TARGET_OOD_RATIO*100}% 비율 맞추기")
    print("-" * 50)
    
    for class_id, coco_cat in CATEGORY_MAPPING.items():
        n_orig = orig_counts.get(class_id, 0)
        if n_orig == 0:
            needed_ood_counts[class_id] = 0
            continue
            
        # 공식: 필요 OOD 개수 = 원본 개수 * (R / (1-R))
        n_ood = math.ceil(n_orig * (TARGET_OOD_RATIO / (1 - TARGET_OOD_RATIO)))
        needed_ood_counts[class_id] = n_ood
        print(f"Class {class_id} ({coco_cat}): 원본 {n_orig}개 -> OOD {n_ood}개 필요")

    # 3. 생성 시작
    coco = COCO(COCO_ANN_FILE)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)
    
    print("\n🚀 데이터셋 생성 시작...")

    for class_id, coco_cat_name in CATEGORY_MAPPING.items():
        target_inst_count = needed_ood_counts.get(class_id, 0)
        if target_inst_count == 0: continue

        catIds = coco.getCatIds(catNms=[coco_cat_name])
        imgIds = coco.getImgIds(catIds=catIds)
        
        # 랜덤성을 위해 셔플 (선택사항)
        import random
        random.shuffle(imgIds)

        current_inst_count = 0 # 현재까지 모은 객체 수
        
        # 진행바 설정
        pbar = tqdm(total=target_inst_count, desc=f"Class {class_id} ({coco_cat_name})")

        for img_id in imgIds:
            # ✨ [핵심 변경] 파일 개수가 아니라, '객체 개수'가 목표를 넘으면 중단
            if current_inst_count >= target_inst_count:
                break

            img_info = coco.loadImgs(img_id)[0]
            
            # 해당 이미지 안의 타겟 카테고리 박스들 가져오기
            annIds = coco.getAnnIds(imgIds=img_id, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)
            
            if not anns: continue
            
            # 이번 이미지에 들어있는 객체 수
            num_instances_in_img = len(anns)
            
            # 파일 저장
            img_filename = img_info['file_name']
            label_filename = img_filename.replace('.jpg', '.txt')
            
            # 이미지 복사
            src_img_path = os.path.join(COCO_IMG_DIR, img_filename)
            dst_img_path = os.path.join(OUTPUT_IMG_DIR, img_filename)
            if not os.path.exists(dst_img_path):
                shutil.copy(src_img_path, dst_img_path)

            # 라벨 작성
            label_path = os.path.join(OUTPUT_LABEL_DIR, label_filename)
            mode = 'a' if os.path.exists(label_path) else 'w'
            
            with open(label_path, mode) as f:
                for ann in anns:
                    yolo_bbox = convert_coco_bbox_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            # ✨ 카운트 증가 (이미지 1장이 아니라, 박스 개수만큼 증가)
            current_inst_count += num_instances_in_img
            pbar.update(num_instances_in_img)

        pbar.close()
        
        if current_inst_count < target_inst_count:
            print(f"⚠️ 경고: {coco_cat_name} 데이터 부족! 목표: {target_inst_count}, 실제확보: {current_inst_count}")

    print("\n✅ 완료! '객체 수(Instance)' 기준으로 비율이 맞춰졌습니다.")

if __name__ == '__main__':
    main()