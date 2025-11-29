import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_yolo_annotations_with_polygons(
    input_images_dir,
    input_labels_dir,
    output_dir,
    class_names=None,
    image_exts=('.jpg', '.jpeg', '.png'),
    figsize=(10, 10)
):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(input_images_dir, f'*{ext}')))

    for img_path in image_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(input_labels_dir, base_name + '.txt')
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # 라벨 읽기
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.axis('off')

        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                # YOLO bbox format: cx, cy, w, h
                cx, cy, bw, bh = coords
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor='lime', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                label = class_names[cls] if class_names else str(cls)
                ax.text(x1, y1 - 5, label, color='white',
                        bbox=dict(facecolor='green', alpha=0.5, edgecolor='none'), fontsize=8)

            elif len(coords) >= 6 and len(coords) % 2 == 0:
                # Polygon format: x1 y1 x2 y2 ... (normalized)
                pts = np.array(coords).reshape(-1, 2)
                pts[:, 0] *= w  # x
                pts[:, 1] *= h  # y
                poly = plt.Polygon(pts, edgecolor='orange', facecolor='none', linewidth=2)
                ax.add_patch(poly)
                label = class_names[cls] if class_names else str(cls)
                ax.text(pts[0, 0], pts[0, 1] - 5, label, color='white',
                        bbox=dict(facecolor='orange', alpha=0.5, edgecolor='none'), fontsize=8)

            else:
                print(f"⚠️ Skipping intrain label in {label_path}: {line}")
                continue

        # 저장
        output_path = os.path.join(output_dir, base_name + '.png')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f'✅ 모든 이미지에 주석 시각화 완료: {output_dir}')
    
draw_yolo_annotations_with_polygons(
    input_images_dir='added_fish_A_ver2/50per/train/images',
    input_labels_dir='added_fish_A_ver2/50per/train/labels',
    output_dir='boxes/added_fish_A_ver2',
    class_names=['crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn'],
    figsize=(8, 8)
)
