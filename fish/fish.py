import os
from collections import defaultdict
from pathlib import Path
import shutil

def rebalance_yolo_labels(input_root, output_root, num_classes=5, max_diff=2):
    input_labels = Path(input_root) / "labels"
    input_images = Path(input_root) / "images"
    output_labels = Path(output_root) / "labels"
    output_images = Path(output_root) / "images"

    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    # 1. 라벨 데이터 로드
    label_data = {}
    class_counts = defaultdict(int)

    for label_file in input_labels.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        label_data[label_file.stem] = lines
        for line in lines:
            cls = int(line.split()[0])
            class_counts[cls] += 1

    def get_major_minor_classes(counts):
        sorted_counts = sorted(counts.items(), key=lambda x: x[1])
        return sorted_counts[0][0], sorted_counts[-1][0]

    # 2. 클래스 균등화
    changed = True
    while changed:
        min_cls, max_cls = get_major_minor_classes(class_counts)
        diff = class_counts[max_cls] - class_counts[min_cls]
        if diff <= max_diff:
            break

        changed = False

        # 이동 시도
        for stem, lines in label_data.items():
            new_lines = []
            modified = False
            for line in lines:
                parts = line.split()
                cls = int(parts[0])
                if cls == max_cls:
                    parts[0] = str(min_cls)
                    class_counts[max_cls] -= 1
                    class_counts[min_cls] += 1
                    modified = True
                    changed = True
                    new_lines.append(" ".join(parts))
                    new_lines += [l for l in lines if l != line]  # 나머지 원래대로
                    break
            if modified:
                label_data[stem] = new_lines
            if changed:
                break

    # 3. 결과 저장
    for stem, lines in label_data.items():
        # 라벨 저장
        out_label_file = output_labels / f"{stem}.txt"
        with open(out_label_file, 'w') as f:
            f.write("\n".join(lines) + "\n")

        # 이미지 복사
        found = False
        for ext in [".jpg", ".jpeg", ".png"]:
            in_img = input_images / f"{stem}{ext}"
            if in_img.exists():
                shutil.copy(in_img, output_images / in_img.name)
                found = True
                break
        if not found:
            print(f"[경고] 이미지 없음: {stem}")

    print("✅ 클래스별 최종 개수:")
    for cls in range(num_classes):
        print(f"Class {cls}: {class_counts[cls]}개")
    print(f"\n✅ 균등화된 결과가 {output_root}에 저장되었습니다.")

# 사용 예시
rebalance_yolo_labels("multiclass_fish", "balance_fish", num_classes=16)