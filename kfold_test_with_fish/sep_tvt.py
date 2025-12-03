import os
import shutil
import random
from collections import defaultdict

def split_dataset_by_class(input_dir, output_dir, seed=42):
    random.seed(seed)

    labels_dir = os.path.join(input_dir, 'labels')
    images_dir = os.path.join(input_dir, 'images')

    # 1. 클래스별 라벨 파일 목록 수집
    class_to_files = defaultdict(list)
    for fname in os.listdir(labels_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(labels_dir, fname)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                class_to_files[cls].append(fname)
                break  # 첫 줄의 클래스만 사용

    # 2. 클래스별 8:1:1 분할
    splits = ['train', 'valid', 'test']
    split_files = {'train': [], 'valid': [], 'test': []}

    for cls, files in class_to_files.items():
        files = list(set(files))  # 중복 제거
        random.shuffle(files)
        n = len(files)
        n_train = int(n * 0.8)
        n_valid = int(n * 0.1)
        n_test = n - n_train - n_valid

        split_files['train'].extend(files[:n_train])
        split_files['valid'].extend(files[n_train:n_train + n_valid])
        split_files['test'].extend(files[n_train + n_valid:])

        print(f"클래스 {cls}: 총 {n}개 → train {n_train}, valid {n_valid}, test {n_test}")

    # 3. 파일 복사
    for split in splits:
        split_label_dir = os.path.join(output_dir, split, 'labels')
        split_image_dir = os.path.join(output_dir, split, 'images')
        os.makedirs(split_label_dir, exist_ok=True)
        os.makedirs(split_image_dir, exist_ok=True)

        for fname in split_files[split]:
            # 라벨 파일 복사
            src_label = os.path.join(labels_dir, fname)
            dst_label = os.path.join(split_label_dir, fname)
            shutil.copy(src_label, dst_label)

            # 이미지 파일 복사
            img_name = os.path.splitext(fname)[0] + '.jpg'
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(split_image_dir, img_name)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"⚠️ 이미지 없음: {src_img}")

    print("\n✅ 데이터 분할 및 복사 완료.")

# 사용 예시
split_dataset_by_class(
    input_dir='merged_A',     # 원본 데이터 (labels/, images/)
    output_dir='A_tvt'       # 나눌 위치 (train/, valid/, test/)
)