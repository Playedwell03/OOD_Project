import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# 디렉토리 경로 설정
label_dir = 'trash_removed/all_removed/labels'
yaml_path = 'merged_data/data.yaml'
output_dir = 'trash_removed/all_removed'  # 저장 디렉토리

# 클래스 이름 불러오기
with open(yaml_path, 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# 라벨 개수 세기 (첫 줄만)
label_counts = Counter()
total = 0

for file_name in os.listdir(label_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(label_dir, file_name)
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                class_index = int(first_line.split()[0])
                label_counts[class_index] += 1
                total += 1

# 결과 정리
counts = [label_counts[i] for i in range(len(class_names))]

# 시각화
plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, counts, color='skyblue')
plt.xlabel('Class')
plt.ylabel('Label Count (1st line only)')
plt.title('Label Count per Class (first line only)', fontsize=14)

# 전체 라벨 수 텍스트 추가 (그래프 내부 오른쪽 위)
plt.text(
    x=len(class_names)-0.5,  # 오른쪽 끝 근처
    y=max(counts) * 0.95,
    s=f'Total: {total}',
    ha='right',
    va='top',
    fontsize=12,
    fontweight='bold',
    color='darkblue'
)

# 막대 위 숫자 표시
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             str(count), ha='center', va='bottom')

plt.tight_layout()

# 저장 폴더 생성 및 저장
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, 'label_count.png')
plt.savefig(save_path)
plt.show()

print(f"✅ 총 라벨 수 (첫 줄 기준): {total}")
print(f"✅ 그래프 저장 완료: {save_path}")