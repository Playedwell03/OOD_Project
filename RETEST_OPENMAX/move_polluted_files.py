import os
import shutil
import random
import yaml
from collections import defaultdict

# 설정
input_dir = 'fish_limited_5class'
output_dir = 'test_data_v3'
yaml_path = 'dataset_v1/data.yaml'
n = 3  # 클래스당 옮길 라벨 수

# 경로
input_labels = os.path.join(input_dir, 'labels')
input_images = os.path.join(input_dir, 'images')
output_labels = os.path.join(output_dir, 'labels')
output_images = os.path.join(output_dir, 'images')

# 디렉토리 생성
os.makedirs(output_labels, exist_ok=True)
os.makedirs(output_images, exist_ok=True)

# 클래스 이름 불러오기
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']
num_classes = len(class_names)

# 클래스별로, 한 줄짜리 파일만 추출
class_to_single_label_files = defaultdict(list)

for file_name in os.listdir(input_labels):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_labels, file_name)
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            if len(lines) == 1:
                class_index = int(lines[0].split()[0])
                class_to_single_label_files[class_index].append(file_name)

# 클래스별로 n개씩 파일 선택
files_to_copy = []

for class_idx in range(num_classes):
    eligible_files = class_to_single_label_files[class_idx]
    selected = random.sample(eligible_files, min(n, len(eligible_files)))
    files_to_copy.extend(selected)

# 복사: 라벨
for file_name in files_to_copy:
    src = os.path.join(input_labels, file_name)
    dst = os.path.join(output_labels, file_name)
    shutil.copy2(src, dst)

# 복사: 이미지
image_extensions = ['.jpg', '.jpeg', '.png']
for file_name in files_to_copy:
    base_name = os.path.splitext(file_name)[0]
    for ext in image_extensions:
        image_path = os.path.join(input_images, base_name + ext)
        if os.path.exists(image_path):
            shutil.copy2(image_path, os.path.join(output_images, base_name + ext))
            break  # 가장 먼저 찾은 확장자 기준 복사