import os
import shutil
from collections import defaultdict
import random
import math

def count_class_labels_across_splits(base_dir, splits):
    class_counts = defaultdict(int)
    for split in splits:
        label_dir = os.path.join(base_dir, split, 'labels')
        for filename in os.listdir(label_dir):
            if filename.endswith('.txt'):
                path = os.path.join(label_dir, filename)
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls = int(parts[0])
                        class_counts[cls] += 1
    return class_counts

def group_ood_by_class(ood_labels_dir):
    class_to_files = defaultdict(list)
    for filename in os.listdir(ood_labels_dir):
        if not filename.endswith('.txt'):
            continue
        path = os.path.join(ood_labels_dir, filename)
        with open(path, 'r') as f:
            lines = f.readlines()
        class_ids = set()
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_ids.add(int(parts[0]))
        if len(class_ids) == 1:
            cls = class_ids.pop()
            class_to_files[cls].append(filename)
    return class_to_files

def add_ood_to_all_splits(base_dataset_dir, ood_dataset_dir, output_base_dir, seed=42):
    random.seed(seed)
    splits = ['train', 'valid', 'test']

    # 경로 설정
    ood_labels_dir = os.path.join(ood_dataset_dir, 'labels')
    ood_images_dir = os.path.join(ood_dataset_dir, 'images')

    # 1. 전체 클래스별 라벨 개수 집계
    class_counts = count_class_labels_across_splits(base_dataset_dir, splits)

    print("📊 전체 클래스별 라벨 개수:")
    for cls, count in sorted(class_counts.items()):
        print(f"  클래스 {cls}: {count}개")

    # 2. 클래스별 사용할 OOD 샘플 결정
    ood_class_to_files = group_ood_by_class(ood_labels_dir)

    cls_to_sampled_ood = {}
    print("\n📦 클래스별 OOD 샘플링:")
    for cls, count in sorted(class_counts.items()):
        needed = math.ceil(count * 0.2)
        available_files = ood_class_to_files.get(cls, [])
        if not available_files:
            print(f"  ⚠️ 클래스 {cls} OOD 없음.")
            continue
        if needed > len(available_files):
            print(f"  ⚠️ 클래스 {cls}: 필요한 {needed}개 중 {len(available_files)}개만 사용.")
            sampled = available_files
        else:
            sampled = random.sample(available_files, needed)
        cls_to_sampled_ood[cls] = sampled
        print(f"  ✅ 클래스 {cls}: {len(sampled)}개 사용")

    # 3. split별로 기존 데이터 + OOD 복사
    for split in splits:
        src_labels_dir = os.path.join(base_dataset_dir, split, 'labels')
        src_images_dir = os.path.join(base_dataset_dir, split, 'images')
        out_labels_dir = os.path.join(output_base_dir, split, 'labels')
        out_images_dir = os.path.join(output_base_dir, split, 'images')
        os.makedirs(out_labels_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)

        print(f"\n🚚 {split.upper()} 데이터 복사 중...")

        # 기존 데이터 복사
        for fname in os.listdir(src_labels_dir):
            if not fname.endswith('.txt'):
                continue
            shutil.copy(os.path.join(src_labels_dir, fname), os.path.join(out_labels_dir, fname))
            img_name = os.path.splitext(fname)[0] + '.jpg'
            img_src = os.path.join(src_images_dir, img_name)
            img_dst = os.path.join(out_images_dir, img_name)
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)

        # 클래스별 OOD 데이터에서 일부 분배하여 복사
        for cls, sampled_list in cls_to_sampled_ood.items():
            portion = len(sampled_list) // 10  # 10% 정도를 각 split에 분배
            if split == 'train':
                selected = sampled_list[:portion * 8]
            elif split == 'valid':
                selected = sampled_list[portion * 8:portion * 9]
            else:  # test
                selected = sampled_list[portion * 9:]

            for fname in selected:
                label_src = os.path.join(ood_labels_dir, fname)
                label_dst = os.path.join(out_labels_dir, fname)
                shutil.copy(label_src, label_dst)

                img_name = os.path.splitext(fname)[0] + '.jpg'
                img_src = os.path.join(ood_images_dir, img_name)
                img_dst = os.path.join(out_images_dir, img_name)
                if os.path.exists(img_src):
                    shutil.copy(img_src, img_dst)
                else:
                    print(f"    ⚠️ 이미지 없음: {img_src}")

    print("\n✅ 전체 Split에 OOD 데이터 추가 완료.")

# 사용 예시
add_ood_to_all_splits(
    base_dataset_dir='splitted_data',      # 기존 train/valid/test 구조
    ood_dataset_dir='multiclass_fish',     # OOD 데이터
    output_base_dir='v2'                   # 출력 디렉토리 (train/valid/test 포함)
)