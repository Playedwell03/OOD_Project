import os
import cv2
import numpy as np


# 🛠 설정
images_dir = "mixed/train/images"     # 이미지가 있는 폴더
labels_dir = "mixed/train/labels"     # YOLO 라벨(.txt) 폴더
output_dir = "boxes"             # 결과 저장 폴더
class_names = ['crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn']

# 📂 출력 폴더 생성
os.makedirs(output_dir, exist_ok=True)

# 이미지 파일 순회
for img_file in os.listdir(images_dir):
    if img_file.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

        # 🖼️ 이미지 로드
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # 📖 라벨 파일 읽기
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    cls_name = class_names[cls_id]

                    coords = list(map(float, parts[1:]))

                    if len(coords) == 4:
                        # 🟩 바운딩 박스 (x_center, y_center, w, h)
                        x_center, y_center, bw, bh = coords
                        x_center, y_center = int(x_center * w), int(y_center * h)
                        bw, bh = int(bw * w), int(bh * h)
                        xmin = int(x_center - bw / 2)
                        ymin = int(y_center - bh / 2)
                        xmax = int(x_center + bw / 2)
                        ymax = int(y_center + bh / 2)

                        # 바운딩 박스 그리기
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        cv2.putText(img, cls_name, (xmin, ymin - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        # 🟥 폴리곤 처리 (x1 y1 x2 y2 ...)
                        points = np.array(coords).reshape(-1, 2)
                        points[:, 0] = points[:, 0] * w  # x 좌표 스케일링
                        points[:, 1] = points[:, 1] * h  # y 좌표 스케일링
                        points = points.astype(np.int32)

                        # 폴리곤 그리기
                        cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=2)
                        # 클래스명 추가
                        x_text, y_text = points[0]
                        cv2.putText(img, cls_name, (x_text, y_text - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            print(f"⚠️ 라벨 파일 없음: {label_path}")

        # 💾 결과 저장
        save_path = os.path.join(output_dir, img_file)
        cv2.imwrite(save_path, img)
        print(f"✅ 저장됨: {save_path}")

print("\n🎉 모든 이미지 시각화 완료!")