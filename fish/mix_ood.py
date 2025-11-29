import os
import random
import shutil
from collections import defaultdict

# 📂 원본 YOLO 라벨 디렉토리와 이미지 디렉토리
label_dir = 'multiclass_fish/labels'
image_dir = 'multiclass_fish/images'

# 📂 결과 저장 디렉토리 (라벨, 이미지 동시 생성)
output_dirs = ['mixed/train', 'mixed/valid']
for out_dir in output_dirs:
    os.makedirs(os.path.join(out_dir, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

# 📝 output1 클래스별 목표 개수 지정 (클래스 16개, 0~15)
target_counts_output1 = {
    0: 268,
    1: 139,
    2: 127,
    3: 173,
    4: 264,
    5: 234,
    6: 174,
    7: 145,
    8: 169,
    9: 112,
    10: 241,
    11: 130,
    12: 581,
    13: 136,
    14: 174,
    15: 189
}

# 📝 output2 클래스별 목표 개수 지정 (클래스 16개, 0~15)
target_counts_output2 = {
    0: 28,
    1: 16,
    2: 18,
    3: 18,
    4: 27,
    5: 27,
    6: 24,
    7: 18,
    8: 21,
    9: 15,
    10: 29,
    11: 18,
    12: 73,
    13: 15,
    14: 21,
    15: 20
}

# 1️⃣ 모든 바운딩박스 로드
all_boxes = []
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        with open(os.path.join(label_dir, filename), 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                box = {
                    'file': filename,
                    'class': int(parts[0]),
                    'coords': parts[1:]  # [x_center, y_center, w, h]
                }
                all_boxes.append(box)

print(f"📦 총 바운딩박스 수: {len(all_boxes)}")

# 2️⃣ output1과 output2를 각각 다른 목표로 랜덤 분배
target_counts_list = [target_counts_output1, target_counts_output2]

for i, (out_dir, target_counts) in enumerate(zip(output_dirs, target_counts_list)):
    new_boxes = []
    class_counts = defaultdict(int)

    boxes_copy = all_boxes.copy()
    random.seed(42 + i)  # output1과 output2가 겹치지 않도록 서로 다른 시드 사용
    random.shuffle(boxes_copy)

    for box in boxes_copy:
        # 아직 목표에 도달하지 못한 클래스들만 선택
        available_classes = [cls for cls, count in target_counts.items()
                             if class_counts[cls] < count]
        if not available_classes:
            break  # 목표 달성 완료
        # 랜덤하게 클래스 선택
        new_class = random.choice(available_classes)
        class_counts[new_class] += 1
        box['class'] = new_class
        new_boxes.append(box)

    print(f"✅ {out_dir}: 클래스별 재분배 완료")
    print(f"📊 {out_dir} 클래스 분포:", dict(class_counts))

    # 3️⃣ 결과를 파일별로 그룹화
    file_groups = defaultdict(list)
    for box in new_boxes:
        file_groups[box['file']].append(box)

    # 4️⃣ 라벨 및 이미지 파일 저장
    for filename, boxes in file_groups.items():
        # ✅ 라벨 저장
        out_label_path = os.path.join(out_dir, 'labels', filename)
        with open(out_label_path, 'w') as f:
            for box in boxes:
                line = f"{box['class']} {' '.join(box['coords'])}\n"
                f.write(line)

        # ✅ 이미지 복사 (확장자 확인: .jpg/.png)
        base_name = os.path.splitext(filename)[0]
        img_extensions = ['.jpg', '.png', '.jpeg']
        found_image = False
        for ext in img_extensions:
            img_file = base_name + ext
            img_path = os.path.join(image_dir, img_file)
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(out_dir, 'images', img_file))
                found_image = True
                break
        if not found_image:
            print(f"⚠️ 이미지 파일 없음: {base_name}")

print("🎉 output1과 output2에 라벨+이미지 저장 완료.")