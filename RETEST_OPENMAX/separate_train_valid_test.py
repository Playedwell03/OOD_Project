import os
import shutil
import random

def split_dataset(labels_dir, images_dir, output_dir, train_count=130, valid_count=16, test_count=16):
    os.makedirs(output_dir, exist_ok=True)

    # 하위 디렉토리 생성
    def make_dirs(base):
        labels_path = os.path.join(base, "labels")
        images_path = os.path.join(base, "images")
        os.makedirs(labels_path, exist_ok=True)
        os.makedirs(images_path, exist_ok=True)
        return labels_path, images_path

    train_labels_dir, train_images_dir = make_dirs(os.path.join(output_dir, "train"))
    valid_labels_dir, valid_images_dir = make_dirs(os.path.join(output_dir, "valid"))
    test_labels_dir, test_images_dir = make_dirs(os.path.join(output_dir, "test"))

    # 클래스별 파일 분류
    file_by_class = {i: [] for i in range(5)}  # 클래스 0~4

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith(".txt"):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, "r") as f:
            lines = f.readlines()
        if not lines:
            continue
        try:
            first_class = int(lines[0].split()[0])
            if 0 <= first_class <= 4:
                file_by_class[first_class].append(label_file)
        except (ValueError, IndexError):
            continue

    # 클래스별 셔플 및 분할
    split_result = {'train': [], 'valid': [], 'test': []}

    for cls in range(5):
        files = file_by_class[cls]
        random.shuffle(files)

        required_total = train_count + valid_count + test_count
        if len(files) < required_total:
            print(f"[경고] 클래스 {cls}의 파일 수가 부족합니다. 총 {len(files)}개 (필요: {required_total})")

        split_result['train'].extend(files[:train_count])
        split_result['valid'].extend(files[train_count:train_count+valid_count])
        split_result['test'].extend(files[train_count+valid_count:train_count+valid_count+test_count])

    # 저장 함수
    def save_split(file_list, label_dst_dir, image_dst_dir):
        for file in file_list:
            label_src = os.path.join(labels_dir, file)
            label_dst = os.path.join(label_dst_dir, file)
            shutil.copy(label_src, label_dst)

            image_name = os.path.splitext(file)[0] + ".jpg"
            image_src = os.path.join(images_dir, image_name)
            image_dst = os.path.join(image_dst_dir, image_name)
            if os.path.exists(image_src):
                shutil.copy(image_src, image_dst)
            else:
                print(f"[경고] 이미지 누락: {image_src}")

    save_split(split_result['train'], train_labels_dir, train_images_dir)
    save_split(split_result['valid'], valid_labels_dir, valid_images_dir)
    save_split(split_result['test'], test_labels_dir, test_images_dir)

    print("✅ Train/Valid/Test 데이터셋 분할 완료!")

# 사용 예시
split_dataset("one_labels_data_final/labels", "one_labels_data_final/images", "data_for_learn2")