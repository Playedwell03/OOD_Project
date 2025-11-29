import os
import shutil
from collections import defaultdict

TARGET_CLASSES = {0, 1}

def clean_and_filter_label_file(input_path, output_path):
    """0~4 클래스만 남기고, 나머지는 제거. 빈줄 없이 저장."""
    with open(input_path, 'r') as f:
        lines = f.readlines()

    # 0~4 클래스만 필터링
    filtered = [line for line in lines if line.strip() and int(line.split()[0]) in TARGET_CLASSES]

    # 저장
    if filtered:
        with open(output_path, 'w') as f:
            f.writelines(filtered)
        return True  # 유효한 라벨 있음
    return False  # 라벨 모두 삭제됨

def copy_file_pair(base_images_dir, base_labels_dir, file_name, output_images_dir, output_labels_dir):
    # 라벨 정리하면서 복사
    input_label_path = os.path.join(base_labels_dir, file_name + '.txt')
    output_label_path = os.path.join(output_labels_dir, file_name + '.txt')

    valid = clean_and_filter_label_file(input_label_path, output_label_path)
    if valid:
        shutil.copy2(os.path.join(base_images_dir, file_name + '.jpg'),
                     os.path.join(output_images_dir, file_name + '.jpg'))
        return True
    else:
        # 라벨이 0~4가 하나도 없으면 이미지도 복사하지 않음
        os.remove(output_label_path)
        return False

def balance_classes(labels_dir, images_dir, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)

    class_to_files = defaultdict(list)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue

        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        classes = [int(line.split()[0]) for line in lines if line.strip()]
        file_classes = set(cls for cls in classes if cls in TARGET_CLASSES)

        for cls in file_classes:
            class_to_files[cls].append(label_file)

    # 균형 맞출 최소 개수
    min_count = min(len(class_to_files[cls]) for cls in TARGET_CLASSES)
    selected_files = set()

    for cls in TARGET_CLASSES:
        selected = class_to_files[cls][:min_count]
        selected_files.update(selected)

    copied = 0
    for label_file in selected_files:
        file_base = os.path.splitext(label_file)[0]
        success = copy_file_pair(images_dir, labels_dir, file_base,
                                 os.path.join(output_dir, 'images'),
                                 os.path.join(output_dir, 'labels'))
        if success:
            copied += 1

    print(f'클래스별 최대 {min_count}개씩 균형 맞게 복사 완료. 유효한 파일 수: {copied}')

# 사용 예시
labels_dir = 'multiclass_fish/labels'
images_dir = 'multiclass_fish/images'
output_dir = 'fish_two_classes'
balance_classes(labels_dir, images_dir, output_dir)