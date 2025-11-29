import os
import shutil
from collections import defaultdict

def filter_labels_and_copy_images_dynamic(
    labels_input_dir,
    images_input_dir,
    labels_output_dir,
    images_output_dir,
    allowed_classes={0,1,2,3,4,5,6,7,8,9,10,11,12}
):
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    saved_class_counts = defaultdict(int)

    label_filenames = [f for f in os.listdir(labels_input_dir) if f.endswith('.txt')]
    print(f"총 라벨 파일 수: {len(label_filenames)}")

    for filename in label_filenames:
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

        # 현재까지 가장 적게 저장된 클래스를 선택
        candidate_classes = list(class_to_lines.keys())
        selected_class = min(candidate_classes, key=lambda c: saved_class_counts[c])
        selected_lines = class_to_lines[selected_class]

        # 저장
        with open(output_label_path, 'w') as f:
            f.writelines(selected_lines)

        saved_class_counts[selected_class] += len(selected_lines)

        # 이미지 복사
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

    print("\n📊 저장된 클래스별 라벨 수:")
    for k in sorted(saved_class_counts):
        print(f"  클래스 {k}: {saved_class_counts[k]}개")

    print(f"\n✅ 완료: 라벨은 {labels_output_dir}, 이미지 복사는 {images_output_dir}에 저장됨.")
    
filter_labels_and_copy_images_dynamic(
    labels_input_dir='fish/fish_merged/labels',
    images_input_dir='fish/fish_merged/images',
    labels_output_dir='one_labels_fish/labels',
    images_output_dir='one_labels_fish/images'
)