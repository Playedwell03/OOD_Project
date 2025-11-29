import os

def count_label_lines(label_dir):
    count = 0
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    for file in label_files:
        with open(os.path.join(label_dir, file), 'r') as f:
            lines = f.readlines()
            count += len([line for line in lines if line.strip()])
    return count

def compare_labels(normal_dir, target_dir, trash_dir):
    def get_label_dict(directory):
        label_dict = {}
        for fname in os.listdir(directory):
            if not fname.endswith('.txt'):
                continue
            path = os.path.join(directory, fname)
            with open(path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                label_dict[fname] = lines
        return label_dict

    normal_labels = get_label_dict(normal_dir)
    trash_labels = get_label_dict(trash_dir)
    target_labels = get_label_dict(target_dir)

    normal_count = 0
    trash_count = 0
    others_count = 0

    for fname, lines in target_labels.items():
        if fname in normal_labels:
            normal_count += len(lines)
        elif fname in trash_labels:
            trash_count += len(lines)
        else:
            others_count += len(lines)

    total = normal_count + trash_count + others_count

    print("비교 결과 (객체 수 기준):")
    print(f"전체 라벨 개수: {total}")
    print(f"정상 데이터 포함: {normal_count}개")
    print(f"쓰레기 데이터 포함: {trash_count}개")
    print(f"OTHERS (둘 다 아닌): {others_count}개")

    return {
        "normal": normal_count,
        "trash": trash_count,
        "others": others_count
    }

# 사용 예시
result = compare_labels(
    normal_dir='merged_data/labels',
    target_dir='final_data_trash_removed_v2/labels',
    trash_dir='multiclass_fish/labels'
)