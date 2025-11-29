import os

def find_multi_label_files(label_dir):
    multi_label_files = []

    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, 'r') as f:
            lines = [line for line in f if line.strip()]
            if len(lines) >= 2:
                multi_label_files.append((fname, len(lines)))

    return multi_label_files

# 사용 예시
label_dir = 'one_labels_data/labels'  # 여기에 본인 라벨 폴더 경로 입력
multi_label_files = find_multi_label_files(label_dir)

print(f"\n📄 2개 이상의 라벨이 포함된 파일 수: {len(multi_label_files)}개")
for fname, count in multi_label_files:
    print(f" - {fname}: {count}개 라벨")


