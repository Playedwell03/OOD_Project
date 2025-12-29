import json
import cv2
import os
import random
import shutil
from tqdm import tqdm

# =========================================================
# [ 사용자 설정 ] 경로를 맞춰주세요
# =========================================================
# 1. 이미지가 들어있는 폴더 (원본 이미지 경로)
IMG_DIR = 'coco_ood_final/images' 

# 2. 확인하고 싶은 JSON 파일 경로
JSON_FILE = 'coco_ood_final/ood_instances.json' # (혹은 변환한 json 파일 경로)

# 3. 결과 이미지를 저장할 폴더 (여기에 박스 그려진 사진이 저장됨)
SAVE_DIR = 'json_box_check_result/coco_cropped'

# 4. 몇 장이나 확인해볼까요?
NUM_SAMPLES = 200
# =========================================================

def visualize_coco(img_dir, json_file, save_dir, num_samples=10):
    # 1. 저장 폴더 초기화
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    # 2. JSON 로드
    print(f"📂 JSON 파일 로딩 중: {json_file}")
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    # 3. 데이터 인덱싱 (이미지 ID로 검색하기 쉽게 변환)
    # 이미지 정보: {image_id: {file_name, width, height}}
    images = {img['id']: img for img in coco_data['images']}
    
    # 카테고리 정보: {category_id: name}
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 어노테이션 정보: {image_id: [ann1, ann2, ...]}
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)

    print(f"📊 총 이미지 수: {len(images)}")
    print(f"📊 총 어노테이션 수: {len(coco_data['annotations'])}")

    # 4. 랜덤 샘플링
    sample_img_ids = random.sample(list(images.keys()), min(num_samples, len(images)))

    print(f"\n🚀 {num_samples}장 랜덤 추출하여 시각화 중...")
    
    for img_id in tqdm(sample_img_ids):
        img_info = images[img_id]
        filename = img_info['file_name']
        
        # 이미지 파일 읽기
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ 이미지 파일 없음: {filename}")
            continue

        # 해당 이미지의 박스 정보 가져오기
        anns = img_to_anns.get(img_id, [])
        
        for ann in anns:
            # COCO bbox format: [x_min, y_min, width, height]
            x, y, w, h = map(int, ann['bbox'])
            category_id = ann['category_id']
            category_name = categories.get(category_id, str(category_id))

            # 박스 그리기 (녹색)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 텍스트 그리기 (클래스 이름)
            text = f"{category_name} ({category_id})"
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 저장
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img)

    print(f"\n✅ 확인 완료! '{save_dir}' 폴더를 확인해보세요.")

if __name__ == "__main__":
    visualize_coco(IMG_DIR, JSON_FILE, SAVE_DIR, NUM_SAMPLES)