import os
from collections import defaultdict
import csv

def compare_images_and_labels(normal_dir, target_dir, trash_dir, target_label_dir, output_path="compare_result"):
    def get_image_names(directory):
        return set(os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    def parse_label_file(label_path):
        class_counts = defaultdict(int)
        if not os.path.exists(label_path):
            return class_counts
        with open(label_path, 'r') as f:
            for line in f:
                if line.strip():
                    cls = line.strip().split()[0]
                    class_counts[cls] += 1
        return class_counts

    os.makedirs(output_path, exist_ok=True)

    normal_set = get_image_names(normal_dir)
    target_set = get_image_names(target_dir)
    trash_set = get_image_names(trash_dir)

    normal_overlap = target_set & normal_set
    trash_overlap = target_set & trash_set
    others = target_set - (normal_set | trash_set)

    result = {
        "normal": list(normal_overlap),
        "trash": list(trash_overlap),
        "others": list(others)
    }

    # 클래스별 통계
    class_stats = {
        "normal": defaultdict(int),
        "trash": defaultdict(int),
        "others": defaultdict(int)
    }

    for category, name_list in result.items():
        for name in name_list:
            label_path = os.path.join(target_label_dir, f"{name}.txt")
            label_counts = parse_label_file(label_path)
            for cls, count in label_counts.items():
                class_stats[category][cls] += count

    # 📊 출력
    print("📊 비교 결과:")
    print(f"✅ 정상 데이터 포함: {len(normal_overlap)}개")
    print(f"🗑️ 쓰레기 데이터 포함: {len(trash_overlap)}개")
    print(f"❓ 어디에도 없는 데이터: {len(others)}개")

    # 📁 결과 저장 - 텍스트
    with open(os.path.join(output_path, "summary.txt"), "w") as f:
        f.write("📊 비교 결과:\n")
        f.write(f"✅ 정상 데이터 포함: {len(normal_overlap)}개\n")
        f.write(f"🗑️ 쓰레기 데이터 포함: {len(trash_overlap)}개\n")
        f.write(f"❓ 어디에도 없는 데이터: {len(others)}개\n")

    # 📁 이미지 목록 저장
    for category in result:
        with open(os.path.join(output_path, f"{category}_images.txt"), "w") as f:
            for name in result[category]:
                f.write(f"{name}\n")

    # 📈 클래스별 통계 CSV 저장
    csv_path = os.path.join(output_path, "class_stats.csv")
    all_classes = sorted(set(class_stats["normal"].keys()) |
                         set(class_stats["trash"].keys()) |
                         set(class_stats["others"].keys()), key=lambda x: int(x))

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "NORMAL", "TRASH", "OTHERS"])
        for cls in all_classes:
            writer.writerow([
                cls,
                class_stats["normal"].get(cls, 0),
                class_stats["trash"].get(cls, 0),
                class_stats["others"].get(cls, 0)
            ])

    return result, class_stats

# 사용 예시
result, class_stats = compare_images_and_labels(
    normal_dir='merged_data/images',    # 정상 이미지
    target_dir='final_data_origin_v2/images',  # 비교 대상 이미지
    trash_dir='multiclass_fish/images', # 쓰레기 이미지
    target_label_dir='final_data_origin_v2/labels',
    output_path='compare_result/_'  # 저장 폴더
)