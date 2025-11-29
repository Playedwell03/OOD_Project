import os
import shutil
from collections import defaultdict

def filter_labels_and_copy_images(
    labels_input_dir,
    images_input_dir,
    labels_output_dir,
    images_output_dir,
    allowed_classes={0,1,2,3,4}
):
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    # 1. 전체 클래스별 라벨 개수 집계
    total_class_counts = defaultdict(int)

    for filename in os.listdir(labels_input_dir):
        if not filename.endswith('.txt'):
            continue

        with open(os.path.join(labels_input_dir, filename), 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                cls = int(parts[0])
                if cls in allowed_classes:
                    total_class_counts[cls] += 1

    if not total_class_counts:
        print("No valid labels found.")
        return

    print("📊 전체 클래스별 라벨 개수:")
    for k in sorted(total_class_counts):
        print(f"  클래스 {k}: {total_class_counts[k]}개")

    # 2. 각 파일 처리
    for filename in os.listdir(labels_input_dir):
        if not filename.endswith('.txt'):
            continue

        input_label_path = os.path.join(labels_input_dir, filename)
        output_label_path = os.path.join(labels_output_dir, filename)

        with open(input_label_path, 'r') as f:
            lines = f.readlines()

        class_to_lines = defaultdict(list)
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                cls = int(parts[0])
                if cls in allowed_classes:
                    class_to_lines[cls].append(line)

        if not class_to_lines:
            continue  # 유효한 클래스 없음

        # 이 파일에 등장한 클래스 중 전체적으로 가장 적게 등장한 클래스만 유지
        candidate_classes = list(class_to_lines.keys())
        least_class = min(candidate_classes, key=lambda c: total_class_counts[c])
        selected_lines = class_to_lines[least_class]

        if selected_lines:
            with open(output_label_path, 'w') as f:
                f.writelines(selected_lines)

            # 이미지 파일 복사
            base_name = os.path.splitext(filename)[0]
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_filename = base_name + ext
                image_input_path = os.path.join(images_input_dir, image_filename)
                if os.path.exists(image_input_path):
                    image_output_path = os.path.join(images_output_dir, image_filename)
                    shutil.copy(image_input_path, image_output_path)
                    found = True
                    break
            if not found:
                print(f"⚠️  이미지 파일 없음: {base_name} (확장자 .jpg/.png 등)")
    
    print(f"\n✅ 완료: 라벨은 {labels_output_dir}, 이미지 복사는 {images_output_dir}에 저장됨.")
    
filter_labels_and_copy_images(
    labels_input_dir='one_labels_data/train/labels',
    images_input_dir='merged_data/images',
    labels_output_dir='one_labels_data_v3/labels',
    images_output_dir='one_labels_data_v3/images'
)