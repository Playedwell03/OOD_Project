import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# 디렉토리 경로 설정
label_dir = 'mixed/train/labels'  # .txt 라벨 파일들이 있는 디렉토리
yaml_path = 'added_fish/50per/data.yaml'     # data.yaml 경로
output_dir = 'mixed/train'                   # 저장 디렉토리
os.makedirs(output_dir, exist_ok=True)  # 디렉토리 없으면 생성

# 클래스 이름 불러오기
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# 라벨 개수 세기
label_counts = Counter()
for file_name in os.listdir(label_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(label_dir, file_name)
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():  # 빈 줄 무시
                    class_index = int(line.split()[0])
                    label_counts[class_index] += 1

# 결과 정리
counts = [label_counts[i] for i in range(len(class_names))]
total_labels = sum(counts)

# 시각화
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Label Count')
plt.title('Label Count per Class')

# 막대 위에 개수 표시
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             str(count), ha='center', va='bottom')

# 전체 라벨 수 표시
plt.text(0.95, 0.95, f'Total labels: {total_labels}', transform=plt.gca().transAxes,
         ha='right', va='top', fontsize=12, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

plt.tight_layout()

# 저장
output_path = os.path.join(output_dir, 'label_distribution2.png')
plt.savefig(output_path)
plt.show()