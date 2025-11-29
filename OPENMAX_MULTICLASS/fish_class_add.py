import os
import random
from collections import defaultdict
import shutil

def remap_top_classes(labels_dir, output_dir, new_classes=[13, 14, 15], samples_per_class=250):
    os.makedirs(output_dir, exist_ok=True)

    class_to_files = defaultdict(list)
    all_filenames = []

    # 1. 클래스별 파일 목록 수집
    for filename in os.listdir(labels_dir):
        if filename.endswith('.txt'):
            path = os.path.join(labels_dir, filename)
            all_filenames.append(filename)
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    cls = int(line.split()[0])
                    class_to_files[cls].append(filename)
                    break  # 첫 줄의 클래스만 고려 (YOLO 방식이라면 대표 클래스 판단용)

    # 2. 가장 많은 클래스 3개 선택
    sorted_classes = sorted(class_to_files.items(), key=lambda x: len(set(x[1])), reverse=True)
    top3_classes = [cls for cls, _ in sorted_classes[:3]]

    print("Top 3 classes:", top3_classes)

    selected_files = set()
    cls_to_selected = {}

    # 3. 각 클래스에서 250개씩 파일 선택
    for i, cls in enumerate(top3_classes):
        files = list(set(class_to_files[cls]))  # 중복 제거
        sampled = random.sample(files, min(samples_per_class, len(files)))
        cls_to_selected[cls] = (sampled, new_classes[i])
        selected_files.update(sampled)

    # 4. 전체 파일 복사 및 클래스 변환 적용
    for filename in all_filenames:
        src_path = os.path.join(labels_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        with open(src_path, 'r') as f:
            lines = f.readlines()

        if filename in selected_files:
            # 이 파일은 클래스 변경 대상
            for cls, (file_list, new_cls) in cls_to_selected.items():
                if filename in file_list:
                    with open(dst_path, 'w') as f_out:
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_cls = int(parts[0])
                            if old_cls == cls:
                                parts[0] = str(new_cls)
                            f_out.write(' '.join(parts) + '\n')
                    break
        else:
            # 그냥 복사
            shutil.copy(src_path, dst_path)

# 예시 실행
remap_top_classes(labels_dir='one_labels_fish/labels', output_dir='multiclass_fish/labels')