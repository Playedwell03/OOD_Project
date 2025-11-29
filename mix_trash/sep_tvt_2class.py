import os
import shutil
import random
from collections import defaultdict

def parse_label_file(label_path):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    class_indices = set(int(line.split()[0]) for line in lines)
    return class_indices

def copy_pair(file_base, img_dir, label_dir, out_img_dir, out_label_dir):
    shutil.copy2(os.path.join(img_dir, file_base + '.jpg'), os.path.join(out_img_dir, file_base + '.jpg'))
    shutil.copy2(os.path.join(label_dir, file_base + '.txt'), os.path.join(out_label_dir, file_base + '.txt'))

def split_dataset(
    img_dir,
    label_dir,
    output_dir,
    train_per_class=100,
    valid_per_class=20,
    test_per_class=30,
    target_classes=(0, 1)
):
    target_classes = set(target_classes)

    # 출력 디렉토리 생성
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 클래스별 단일 클래스 라벨 파일 수집
    class_to_files = defaultdict(list)
    for label_file in os.listdir(label_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(label_dir, label_file)
        class_indices = parse_label_file(label_path)

        if len(class_indices) == 1:
            cls = next(iter(class_indices))
            if cls in target_classes:
                class_to_files[cls].append(label_file)

    # 클래스별로 섞기
    for cls in target_classes:
        random.shuffle(class_to_files[cls])

    used_files = set()

    # 분할 및 복사
    for split, per_class in [('train', train_per_class), ('valid', valid_per_class), ('test', test_per_class)]:
        for cls in target_classes:
            candidates = [f for f in class_to_files[cls] if f not in used_files]
            selected = candidates[:per_class]
            if len(selected) < per_class:
                print(f'⚠️ 클래스 {cls}의 {split} 셋에 필요한 {per_class}개 중 {len(selected)}개만 확보됨.')
            for label_file in selected:
                used_files.add(label_file)
                base = os.path.splitext(label_file)[0]
                copy_pair(base, img_dir, label_dir,
                          os.path.join(output_dir, split, 'images'),
                          os.path.join(output_dir, split, 'labels'))

    print('✅ 라벨 기준으로 데이터셋 분할 완료.')

# 실행 예시
split_dataset(
    img_dir='A_two_classes_final/images',
    label_dir='A_two_classes_final/labels',
    output_dir='A_5',
    train_per_class=130,
    valid_per_class=15,
    test_per_class=15
)