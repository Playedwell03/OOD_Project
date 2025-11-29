import os
import cv2
import matplotlib.pyplot as plt

# 바운딩 박스를 시각화하는 함수
def draw_boxes_on_image(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    if not os.path.exists(label_path):
        return image  # 라벨 없으면 원본 그대로 반환

    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 형식에 맞지 않는 줄 무시
            class_id, x_center, y_center, box_w, box_h = map(float, parts[:5])
            x1 = int((x_center - box_w / 2) * w)
            y1 = int((y_center - box_h / 2) * h)
            x2 = int((x_center + box_w / 2) * w)
            y2 = int((y_center + box_h / 2) * h)

            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = class_names[int(class_id)] if int(class_id) < len(class_names) else str(int(class_id))
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image

# 메인 함수
def visualize_dataset_with_boxes(base_dir, output_dir, class_names):
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)

    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        output_path = os.path.join(output_dir, img_name)

        img_with_boxes = draw_boxes_on_image(img_path, label_path, class_names)
        cv2.imwrite(output_path, img_with_boxes)

    print(f"시각화 완료: {output_dir}에 저장됨")

# 사용 예시
if __name__ == "__main__":
    base_dir = "multiclass_fish"  # 여기에 사용자 디렉토리 입력
    output_dir = "box"
    class_names = ['crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn']  # 클래스 이름

    visualize_dataset_with_boxes(base_dir, output_dir, class_names)