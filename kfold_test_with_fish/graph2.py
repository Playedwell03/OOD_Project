import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os # 폴더 생성을 위해 os 모듈 추가

# 1. YOLO 훈련 결과 파일 경로
results_path = 'models/A/results.csv' 

# ✨ 1-1. 그래프를 저장할 폴더 및 파일 이름 설정
output_dir = 'result/A'
output_filename = 'metrics.png'
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
        'metrics/precision(B)': np.random.uniform(0.6, 0.9, epochs),
        'metrics/recall(B)': np.random.uniform(0.2, 0.5, epochs),
        'metrics/mAP50(B)': np.random.uniform(0.15, 0.35, epochs),
        'metrics/mAP50-95(B)': np.random.uniform(0.1, 0.27, epochs),
    }
    df = pd.DataFrame(data)

# 3. 그릴 Metrics 컬럼 목록 정의
metrics_columns = [
    'metrics/precision(B)', 
    'metrics/recall(B)', 
    'metrics/mAP50(B)', 
    'metrics/mAP50-95(B)'
]

# 4. 그래프 그리기 (2행 2열)
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

# 5. 각 Metrics 컬럼에 대해 그래프 그리기
for i, col in enumerate(metrics_columns):
    if col in df.columns:
        axes[i].plot(df['epoch'], df[col], marker='o', linestyle='-')
        axes[i].set_title(col, fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Score')
        axes[i].set_ylim(0, 1) # y축 범위를 0에서 1로 고정
        axes[i].grid(True)
    else:
        axes[i].set_title(f"'{col}' not found", fontsize=12)
        axes[i].set_xlim(0, len(df['epoch'])-1 if 'epoch' in df else 9)
        axes[i].set_ylim(0, 1)
        axes[i].grid(True)

# 전체 레이아웃 조정
plt.tight_layout()

# ✨ 6. 그래프를 파일로 저장
plt.savefig(output_path, dpi=300)
print(f"그래프가 '{output_path}' 경로에 저장되었습니다.")

# 7. 화면에 그래프 출력
plt.show()