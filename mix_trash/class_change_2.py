import os
import random
import shutil

def make_ood_dataset(input_dir, output_dir, ood_ratio=0.1, original_class_count=16):
    label_input_dir = os.path.join(input_dir, 'labels')
    image_input_dir = os.path.join(input_dir, 'images')
    label_output_dir = os.path.join(output_dir, 'labels')
    image_output_dir = os.path.join(output_dir, 'images')

    os.makedirs(label_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    all_label_files = [f for f in os.listdir(label_input_dir) if f.endswith('.txt')]
    label_lines = []

    # 전체 라벨 수 세기
    for file in all_label_files:
        with open(os.path.join(label_input_dir, file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = int(line.strip().split()[0])
                label_lines.append((file, label))

    total_labels = len(label_lines)
    ood_target_count = int(total_labels * ood_ratio)
    print(f"[INFO] 전체 라벨 수: {total_labels}, 생성할 OOD 수: {ood_target_count}")

    # OOD 대상 무작위 선택
    ood_label_indices = set(random.sample(range(total_labels), ood_target_count))
    ood_count = 0

    # 파일 별로 다시 처리
    file_line_map = {}
    current_idx = 0  # 전체 인덱스 관리

    for file in all_label_files:
        label_path = os.path.join(label_input_dir, file)
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        used_classes = set(int(line.split()[0]) for line in lines)
        available_classes = [i for i in range(original_class_count) if i not in used_classes]

        new_lines = []
        for line in lines:
            parts = line.split()
            label = int(parts[0])

            if current_idx in ood_label_indices and available_classes:
                new_class = random.choice(available_classes)
                available_classes.remove(new_class)
                parts[0] = str(new_class)
                ood_count += 1

            new_lines.append(' '.join(parts))
            current_idx += 1

        # 저장
        with open(os.path.join(label_output_dir, file), 'w') as f:
            f.write('\n'.join(new_lines))

        # 이미지 복사
        image_file = file.replace('.txt', '.jpg')
        src_image = os.path.join(image_input_dir, image_file)
        dst_image = os.path.join(image_output_dir, image_file)
        if os.path.exists(src_image):
            shutil.copy(src_image, dst_image)

    print(f"[DONE] OOD 라벨 개수: {ood_count}")
    
make_ood_dataset(
    input_dir='splitted/train',
    output_dir='splitted_class_ch/30per/train',
    ood_ratio=0.3,  # 전체 라벨 중 10%를 OOD로 바꿈
    original_class_count=16
)