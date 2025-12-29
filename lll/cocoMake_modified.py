import os
import json
import cv2
import numpy as np
from glob import glob

# 사용자 경로 설정
root_dir = "A_tvt"

classes = [
    'crackdown', 
    'crossline', 
    'danger', 
    'kidprotectzone', 
    'leftcurve', 
    'noparking', 
    'nouturn', 
    'ntrack', 
    'oneway', 
    'slipper', 
    'slow', 
    'speedbump', 
    'speedlimit', 
    'stop', 
    'turncross', 
    'uturn'
]

def yolo_to_coco(img_dir, label_dir, output_json, classes):
    images, annotations, categories = [], [], []
    ann_id = 1

    # Categories 등록
    for i, cls in enumerate(classes):
        categories.append({"id": i, "name": cls, "supercategory": "none"})

    # 이미지 파일 리스트 가져오기
    img_files = sorted(
        glob(os.path.join(img_dir, "**", "*.png"), recursive=True) +
        glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True)
    )

    print(f" Found {len(img_files)} images in {img_dir}")

    for img_id, img_path in enumerate(img_files):
        filename = os.path.basename(img_path)
        
        # 이미지 읽기 (OpenCV)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        h, w = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": w,
            "height": h
        })

        # 라벨 파일 경로 찾기
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir, label_filename)
        
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue

            cls_id = int(parts[0])
            segmentation = []
            bbox = []
            area = 0.0

            # ---------------------------------------------------------
            # CASE 1: Polygon (Segmentation) - 데이터 개수가 5개 초과
            # 형식: <class> <x1> <y1> <x2> <y2> ...
            # ---------------------------------------------------------
            if len(parts) > 5:
                points = []
                # 좌표 파싱 (짝수로 묶어서 처리)
                for i in range(1, len(parts), 2):
                    px = float(parts[i]) * w
                    py = float(parts[i+1]) * h
                    points.append([px, py])
                
                # COCO segmentation은 [[x1, y1, x2, y2, ...]] 형태 (flatten)
                poly_flat = [coord for point in points for coord in point]
                segmentation.append(poly_flat)

                # Polygon에서 Bbox 추출 (min_x, min_y, width, height)
                np_points = np.array(points, dtype=np.float32)
                x_min = float(np.min(np_points[:, 0]))
                y_min = float(np.min(np_points[:, 1]))
                x_max = float(np.max(np_points[:, 0]))
                y_max = float(np.max(np_points[:, 1]))
                
                b_w = x_max - x_min
                b_h = y_max - y_min
                bbox = [x_min, y_min, b_w, b_h]

                # 면적 계산 (Green's theorem / Shoelace formula 대신 OpenCV 활용)
                # cv2.contourArea는 int/float32 numpy array 필요
                area = float(cv2.contourArea(np_points.astype(np.float32)))

            # ---------------------------------------------------------
            # CASE 2: Bounding Box (Detection) - 데이터 개수가 정확히 5개
            # 형식: <class> <xc> <yc> <w> <h>
            # ---------------------------------------------------------
            elif len(parts) == 5:
                xc, yc, bw, bh = map(float, parts[1:5])
                
                # YOLO (center) -> COCO (top-left) 변환
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                b_w = bw * w
                b_h = bh * h
                
                bbox = [x, y, b_w, b_h]
                area = b_w * b_h
                segmentation = [] # Box만 있는 경우 segmentation은 비움

            # Annotation 추가
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls_id,
                "bbox": bbox,
                "segmentation": segmentation,
                "area": area,
                "iscrowd": 0
            }
            annotations.append(ann)
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    # Numpy 타입이 json dump시 에러나지 않도록 처리된 데이터 저장
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2, ensure_ascii=False)
    print(f" Saved: {output_json} (Images: {len(images)}, Annotations: {len(annotations)})")

# 실행 부분
splits = ["train", "valid", "test"]
for split in splits:
    img_dir = os.path.join(root_dir, split, "images")
    label_dir = os.path.join(root_dir, split, "labels")
    output_json = os.path.join(root_dir, f"{split}.json")
    
    # 폴더가 실제로 존재할 때만 실행
    if os.path.exists(img_dir) and os.path.exists(label_dir):
        yolo_to_coco(img_dir, label_dir, output_json, classes)
    else:
        print(f"Skipping {split}: directory not found.")