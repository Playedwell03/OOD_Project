import matplotlib.pyplot as plt
import os

# 데이터
categories = ["Normal", "Trash_1", "Trash_2", "Trash_3", "Trash_4", "Trash_5"]
precision = [94.1, 94.6, 95.6, 95.0, 93.5, 93.6]
recall = [94.0, 96.0, 97.1, 96.7, 96.2, 96.2]
mAP50 = [96.7, 97.0, 97.5, 96.7, 96.8, 96.7]
mAP50_95 = [77.0, 77.6, 78.2, 77.0, 77.2, 77.7]

# output 디렉토리에 저장
output_dir = "img_output"
os.makedirs(output_dir, exist_ok=True)  # 폴더 없으면 생성


# Precision 그래프
plt.figure(figsize=(8, 6))
plt.plot(categories, precision, marker='o', linestyle='-', color='blue')
plt.title('Precision per Category')
plt.ylabel('Precision (%)')
plt.ylim(90, 100)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/precision.png")
plt.close()

# Recall 그래프
plt.figure(figsize=(8, 6))
plt.plot(categories, recall, marker='o', linestyle='-', color='green')
plt.title('Recall per Category')
plt.ylabel('Recall (%)')
plt.ylim(90, 100)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/recall.png")
plt.close()

# mAP50 그래프
plt.figure(figsize=(8, 6))
plt.plot(categories, mAP50, marker='o', linestyle='-', color='red')
plt.title('mAP@50 per Category')
plt.ylabel('mAP@50 (%)')
plt.ylim(90, 100)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/mAP50.png")
plt.close()

# mAP50-95 그래프
plt.figure(figsize=(8, 6))
plt.plot(categories, mAP50_95, marker='o', linestyle='-', color='purple')
plt.title('mAP@50-95 per Category')
plt.ylabel('mAP@50-95 (%)')
plt.ylim(75, 80)
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/mAP50-95.png")
plt.close()