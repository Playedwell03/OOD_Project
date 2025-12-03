import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # 폴더 생성을 위해 os 모듈 추가

# 1. YOLO 훈련 결과 파일 경로
results_path = 'models/A/results.csv' 
# ✨ 1-1. 그래프를 저장할 폴더 및 파일 이름 설정
output_dir = 'result/A'
output_filename = 'loss.png'
output_path = os.path.join(output_dir, output_filename)

# ✨ 1-2. 저장할 폴더가 없으면 자동으로 생성
os.makedirs(output_dir, exist_ok=True)

# 2. CSV 파일 읽기
try:
    # CSV 파일의 공백 제거 및 컬럼 이름 클리닝
    df = pd.read_csv(results_path)
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print(f"오류: '{results_path}' 경로에 파일이 없습니다. 예시 데이터로 실행합니다.")
    # 예시 실행을 위해 임의의 데이터프레임 생성
    epochs = 50
    data = {
        'epoch': range(epochs),
        'train/box_loss': np.linspace(2.0, 0.7, epochs),
        'train/cls_loss': np.linspace(7.0, 3.0, epochs) * np.random.uniform(0.8, 1.2, epochs),
        'val/box_loss': np.linspace(1.45, 1.39, epochs) * np.random.uniform(0.95, 1.05, epochs),
        'val/cls_loss': np.linspace(3.5, 3.28, epochs) * np.random.uniform(0.98, 1.02, epochs),
    }
    df = pd.DataFrame(data)

# 3. 비교하기 좋게 컬럼 순서 재정의
loss_columns = [
    'train/box_loss', 'val/box_loss', 
    'train/cls_loss', 'val/cls_loss'
]

# 4. 그래프 그리기 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# 5. 각 컬럼에 대해 그래프 그리기
for i, col in enumerate(loss_columns):
    if col in df.columns:
        axes[i].plot(df['epoch'], df[col], marker='o', linestyle='-')
        axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].grid(True)

# 6. 같은 종류의 Loss끼리 Y축 범위 통일
# box_loss 스케일 통일
box_cols = ['train/box_loss', 'val/box_loss']
if all(c in df.columns for c in box_cols):
    max_box_loss = df[box_cols].max().max()
    axes[0].set_ylim(0, max_box_loss * 1.1)
    axes[1].set_ylim(0, max_box_loss * 1.1)

# cls_loss 스케일 통일
cls_cols = ['train/cls_loss', 'val/cls_loss']
if all(c in df.columns for c in cls_cols):
    max_cls_loss = df[cls_cols].max().max()
    axes[2].set_ylim(0, max_cls_loss * 1.1)
    axes[3].set_ylim(0, max_cls_loss * 1.1)

# 전체 레이아웃 조정
plt.tight_layout()

# ✨ 7. 그래프를 파일로 저장
plt.savefig(output_path, dpi=300) # dpi 옵션으로 해상도 조절 가능
print(f"그래프가 '{output_path}' 경로에 저장되었습니다.")

# 8. 화면에 그래프 출력
plt.show()