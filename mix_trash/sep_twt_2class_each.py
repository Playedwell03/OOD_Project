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
    # 이미지 확장자를 '.jpg'로 고정하지 않고, 원본 폴더에서 찾아 동적으로 처리
    img_ext = '.jpg' # 기본 확장자
    for ext in ['.jpg', '.jpeg', '.png']:
        if os.path.exists(os.path.join(img_dir, file_base + ext)):
            img_ext = ext
            break
            
    shutil.copy2(os.path.join(img_dir, file_base + img_ext), os.path.join(out_img_dir, file_base + img_ext))
    shutil.copy2(os.path.join(label_dir, file_base + '.txt'), os.path.join(out_label_dir, file_base + '.txt'))

def split_dataset(
    img_dir,
    label_dir,
    output_dir,
    # 딕셔너리를 받도록 파라미터 변경
    train_counts,
    valid_counts,
    test_counts,
    target_classes=(0, 1)
):
    target_classes = set(target_classes)

    # 출력 디렉토리 생성
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

    # 클래스별 단일 클래스 라벨 파일 수집
    class_to_files = defaultdict(list)
    print("🔍 라벨 파일을 분석 중입니다...")
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
        print(f"  - 클래스 {cls}: 단일 라벨 파일 {len(class_to_files[cls])}개 발견")

    used_files = set()

    # 분할 및 복사 로직 수정
    # 각 분할(split)에 해당하는 딕셔너리를 함께 사용
    for split, counts_dict in [('train', train_counts), ('valid', valid_counts), ('test', test_counts)]:
        print(f"\n🚀 '{split}' 데이터셋 생성 중...")
        for cls in target_classes:
            # 딕셔너리에서 현재 클래스(cls)에 해당하는 개수를 가져옴
            # 만약 키가 없으면 0을 기본값으로 사용
            num_to_select = counts_dict.get(cls, 0)
            
            if num_to_select == 0:
                continue

            candidates = [f for f in class_to_files[cls] if f not in used_files]
            selected = candidates[:num_to_select]
            
            if len(selected) < num_to_select:
                print(f'  ⚠️ 클래스 {cls}: 요청 {num_to_select}개 중 {len(selected)}개만 확보 가능.')
            else:
                 print(f"  - 클래스 {cls}: {len(selected)}개 파일 처리 중...")


            for label_file in selected:
                used_files.add(label_file)
                base = os.path.splitext(label_file)[0]
                copy_pair(base, img_dir, label_dir,
                          os.path.join(output_dir, split, 'images'),
                          os.path.join(output_dir, split, 'labels'))

    print('\n✅ 데이터셋 분할 완료.')

# ===================================================================
# 실행 예시: 각 클래스별로 원하는 개수를 딕셔너리 형태로 지정합니다.
# ===================================================================

# 🎯 분할할 클래스를 지정합니다.
target_classes_to_split = (0, 1)

# 🔢 클래스별 train 개수를 지정합니다.
# 예: 클래스 0은 130개, 클래스 1은 150개
train_counts_per_class = {
    0: 232,
    1: 139
}

# 🔢 클래스별 validation 개수를 지정합니다.
# 예: 클래스 0은 15개, 클래스 1은 20개
valid_counts_per_class = {
    0: 28,
    1: 17
}

# 🔢 클래스별 test 개수를 지정합니다.
# 예: 클래스 0은 15개, 클래스 1은 20개
test_counts_per_class = {
    0: 30,
    1: 17
}


split_dataset(
    img_dir='fish_two_classes/images',
    label_dir='fish_two_classes/labels',
    output_dir='added_fish_A6/50per',
    # 위에서 정의한 딕셔너리를 인자로 전달
    train_counts=train_counts_per_class,
    valid_counts=valid_counts_per_class,
    test_counts=test_counts_per_class,
    target_classes=target_classes_to_split
)