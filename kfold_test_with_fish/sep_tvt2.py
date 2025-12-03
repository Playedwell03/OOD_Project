import os
import shutil
import random
from collections import defaultdict
import glob

def create_stratified_k10_folds(input_dir, output_dir, k=10, seed=42):
    random.seed(seed)

    labels_dir = os.path.join(input_dir, 'labels')
    images_dir = os.path.join(input_dir, 'images')

    if not os.path.exists(labels_dir):
        print(f"⚠️ 'labels' 폴더를 찾을 수 없습니다: {labels_dir}")
        return
    if not os.path.exists(images_dir):
        print(f"⚠️ 'images' 폴더를 찾을 수 없습니다: {images_dir}")
        return

    # 1. 클래스별 라벨 파일 목록 수집
    class_to_files = defaultdict(list)
    # (참고: 원본 코드와 동일하게, .txt 파일의 '첫 번째' 객체를 기준으로 클래스를 할당합니다)
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

    print("--- 클래스별 파일 수집 완료 ---")
    for cls, files in class_to_files.items():
        print(f"클래스 {cls}: 총 {len(files)}개")

    # 2. 클래스별로 K=10개 폴더에 분배
    #    (예: 100개 파일 -> 10개씩 10개 폴더)
    #    (예: 53개 파일 -> 6개씩 3개 폴더, 5개씩 7개 폴더)
    
    # K개의 폴더를 나타내는 리스트를 생성
    # folds_files[0] -> 1번 폴더에 들어갈 파일 목록
    # folds_files[1] -> 2번 폴더에 들어갈 파일 목록
    folds_files = [[] for _ in range(k)]

    for cls, files in class_to_files.items():
        files = list(set(files))  # 중복 제거
        random.shuffle(files)
        
        # (중요) 라운드 로빈 방식으로 파일을 K개 폴더에 순차적으로 배분
        # 이렇게 하면 클래스 비율이 모든 폴더에 거의 동일하게 유지됩니다.
        for i, fname in enumerate(files):
            fold_index = i % k
            folds_files[fold_index].append(fname)

    print("\n--- K=10 계층화 분할 완료 ---")

    # 3. 파일 복사
    for i in range(k):
        fold_num = i + 1
        fold_name = f'fold_{fold_num}'
        print(f"Processing {fold_name}...")

        split_label_dir = os.path.join(output_dir, fold_name, 'labels')
        split_image_dir = os.path.join(output_dir, fold_name, 'images')
        os.makedirs(split_label_dir, exist_ok=True)
        os.makedirs(split_image_dir, exist_ok=True)

        file_list = folds_files[i]
        for fname in file_list:
            # 라벨 파일 복사
            src_label = os.path.join(labels_dir, fname)
            dst_label = os.path.join(split_label_dir, fname)
            shutil.copy(src_label, dst_label)

            # 이미지 파일 복사 (확장자를 .jpg로 가정)
            # (만약 .png 등 다른 확장자도 있다면 원본 코드의 로직을 사용해야 합니다)
            img_name = os.path.splitext(fname)[0] + '.jpg' 
            src_img = os.path.join(images_dir, img_name)
            dst_img = os.path.join(split_image_dir, img_name)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                # .jpg가 없다면 .png 등 다른 확장자를 찾아봅니다.
                found = False
                for ext in ['.png', '.jpeg', '.bmp']:
                    img_name_alt = os.path.splitext(fname)[0] + ext
                    src_img_alt = os.path.join(images_dir, img_name_alt)
                    if os.path.exists(src_img_alt):
                        dst_img_alt = os.path.join(split_image_dir, img_name_alt)
                        shutil.copy(src_img_alt, dst_img_alt)
                        found = True
                        break
                if not found:
                     print(f"⚠️ 이미지 없음: {src_img} (및 기타 확장자)")

    print(f"\n✅ {k}-Fold 계층화 데이터 분할 및 복사 완료.")
    print(f"데이터가 '{output_dir}'에 fold_1 ~ fold_{k} 폴더로 저장되었습니다.")

# --- 사용 예시 ---
create_stratified_k10_folds(
    input_dir='merged_A',     # 원본 데이터 (labels/, images/)
    output_dir='A_k10_folds'  # 10개 폴더(fold_1, ..., fold_10)가 생성될 위치
)