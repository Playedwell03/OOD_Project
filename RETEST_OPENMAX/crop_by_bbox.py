import os
import cv2

def crop_yolo_objects(images_dir, labels_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file)[0] + '.txt')

        if not os.path.exists(label_path):
            continue  # 라벨 파일이 없으면 스킵

        img = cv2.imread(image_path)
        h, w, _ = img.shape

        with open(label_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            class_id, x_center, y_center, box_width, box_height = map(float, parts)
            x_center, y_center = x_center * w, y_center * h
            box_width, box_height = box_width * w, box_height * h
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # 좌표 보정
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cropped = img[y1:y2, x1:x2]

            # 출력 파일명: 원본이름 그대로 사용, 객체가 여러 개일 경우 인덱스 붙임
            base_name = os.path.splitext(image_file)[0]
            ext = os.path.splitext(image_file)[1]
            output_filename = f"{base_name}_{i}{ext}" if len(lines) > 1 else f"{base_name}{ext}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped)

    print(f"✅ 모든 객체가 {output_dir}에 저장되었습니다.")
    
crop_yolo_objects(
    images_dir='dataset/train/images',
    labels_dir='dataset/train/labels',
    output_dir='cropped_data/train/images'
)