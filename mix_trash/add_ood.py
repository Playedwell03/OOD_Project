import os
import shutil
from collections import defaultdict
import random
import math

def count_class_labels(base_dir, split):
    label_dir = os.path.join(base_dir, split, 'labels')
    class_counts = defaultdict(int)
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls = int(parts[0])
                        class_counts[cls] += 1
    return class_counts

def group_ood_by_class(ood_labels_dir):
    class_to_files = defaultdict(list)
    for filename in os.listdir(ood_labels_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(ood_labels_dir, filename), 'r') as f:
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

def add_ood_to_train_valid(base_dataset_dir, ood_dataset_dir, output_base_dir, ood_ratio=0.1, seed=42):
    random.seed(seed)
    splits = ['train']

    ood_labels_dir = os.path.join(ood_dataset_dir, 'labels')
    ood_images_dir = os.path.join(ood_dataset_dir, 'images')
    ood_class_to_files = group_ood_by_class(ood_labels_dir)
    used_ood_files = set()

    for split in splits:
        print(f"\n🔄 처리 중: {split.upper()}")

        # 클래스별 원본 데이터 수 세기
        class_counts = count_class_labels(base_dataset_dir, split)

        # 출력 경로 설정
        out_labels_dir = os.path.join(output_base_dir, split, 'labels')
        out_images_dir = os.path.join(output_base_dir, split, 'images')
        os.makedirs(out_labels_dir, exist_ok=True)
        os.makedirs(out_images_dir, exist_ok=True)

        # 기존 데이터 복사
        src_labels_dir = os.path.join(base_dataset_dir, split, 'labels')
        src_images_dir = os.path.join(base_dataset_dir, split, 'images')
        for fname in os.listdir(src_labels_dir):
            if fname.endswith('.txt'):
                shutil.copy(os.path.join(src_labels_dir, fname), os.path.join(out_labels_dir, fname))
                img_name = os.path.splitext(fname)[0] + '.jpg'
                img_src = os.path.join(src_images_dir, img_name)
                img_dst = os.path.join(out_images_dir, img_name)
                if os.path.exists(img_src):
                    shutil.copy(img_src, img_dst)

        # OOD 데이터 샘플링 및 복사
        for cls, count in sorted(class_counts.items()):
            # 총 데이터 대비 OOD 비율이 ood_ratio가 되도록 계산
            num_ood_needed = math.ceil(count * ood_ratio / (1 - ood_ratio))
            available_files = [f for f in ood_class_to_files.get(cls, []) if f not in used_ood_files]
            if len(available_files) < num_ood_needed:
                print(f"  ⚠️ 클래스 {cls}: 필요한 {num_ood_needed}개 중 {len(available_files)}개만 사용")
                selected_files = available_files
            else:
                selected_files = random.sample(available_files, num_ood_needed)

            print(f"  ✅ 클래스 {cls}: {len(selected_files)}개 OOD 추가")

            for fname in selected_files:
                used_ood_files.add(fname)

                # 라벨 복사
                label_src = os.path.join(ood_labels_dir, fname)
                label_dst = os.path.join(out_labels_dir, fname)
                shutil.copy(label_src, label_dst)

                # 이미지 복사
                img_name = os.path.splitext(fname)[0] + '.jpg'
                img_src = os.path.join(ood_images_dir, img_name)
                img_dst = os.path.join(out_images_dir, img_name)
                if os.path.exists(img_src):
                    shutil.copy(img_src, img_dst)
                else:
                    print(f"    ⚠️ 이미지 없음: {img_src}")

    print("\n✅ OOD 추가 완료 (Train/Valid에 중복 없음)")

# 사용 예시
add_ood_to_train_valid(
    base_dataset_dir='A_tvt',
    ood_dataset_dir='multiclass_fish',
    output_base_dir='added_fish_A/50per',
    ood_ratio=0.5  # 전체 기준 10%가 OOD가 되도록
)