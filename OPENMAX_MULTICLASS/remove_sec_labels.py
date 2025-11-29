import os

def keep_only_first_line(label_dir):
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        path = os.path.join(label_dir, fname)
        with open(path, 'r') as f:
            first_line = f.readline()
        if first_line.strip():  # 비어있지 않다면
            with open(path, 'w') as f:
                f.write(first_line)
        else:
            # 첫 줄이 비어 있다면 파일을 비워버림
            open(path, 'w').close()

# 사용 예시
label_dir = 'one_labels_data/labels'  # 라벨 폴더 경로로 변경
keep_only_first_line(label_dir)
print("✅ 모든 라벨 파일에서 첫 줄만 남겼습니다.")