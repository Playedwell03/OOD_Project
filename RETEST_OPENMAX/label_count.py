import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# 디렉토리 경로 설정
label_dir = 'test_data_v3/labels'
yaml_path = 'one_labels_data_final/data.yaml'

# 클래스 이름 불러오기
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# 라벨 개수 세기 (첫 줄만)
label_counts = Counter()

for file_name in os.listdir(label_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(label_dir, file_name)
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_index = int(first_line.split()[0])
                label_counts[class_index] += 1

# 결과 정리
counts = [label_counts[i] for i in range(len(class_names))]

# 시각화
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Label Count (1st line only)')
plt.title('Label Count per Class (first line only)')

# 막대 위 숫자 표시
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()