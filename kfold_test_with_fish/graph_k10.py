import os
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. ⚠️ 사용자 설정 ---

# 10개의 Run 폴더가 들어있는 부모 프로젝트 디렉토리
# (예: 'models_k10_runs_30per', 'models_k10_runs_10per' 등)
PROJECT_DIR = 'models_k10_runs_70per' 

# 결과(평균 CSV, 평균 그래프)를 저장할 폴더
OUTPUT_DIR = 'k10_model_graphs_70per'

K = 10  # K-Fold 횟수
CSV_NAME = 'results.csv' # YOLOv8이 생성하는 CSV 파일 이름
# ---

print(f"--- K={K} Fold Average Graph Generator ---")
print(f"Target Project: {PROJECT_DIR}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 10개의 'results.csv' 파일 경로 찾기 ---
csv_paths = []
for i in range(1, K + 1):
    run_name = f'run_{i}_test'
    path = os.path.join(PROJECT_DIR, run_name, CSV_NAME)
    
    if os.path.exists(path):
        csv_paths.append(path)
    else:
        print(f"⚠️ 경고: '{path}'를 찾을 수 없습니다. 이 Run은 평균 계산에서 제외됩니다.")

if not csv_paths:
    print(f"❌ 오류: '{PROJECT_DIR}'에서 '{CSV_NAME}' 파일을 하나도 찾을 수 없습니다. 스크립트를 종료합니다.")
    exit()

print(f"✅ {len(csv_paths)}개의 'results.csv' 파일을 찾았습니다.")

# --- 3. 모든 CSV 파일을 Pandas로 읽고 하나로 합치기 ---
all_dfs = []
for path in csv_paths:
    df = pd.read_csv(path)
    # (중요) YOLOv8 CSV는 컬럼 이름에 공백이 많습니다. 공백 제거.
    df.columns = df.columns.str.strip() 
    all_dfs.append(df)

# 10개의 DataFrame을 세로로 모두 연결 (Epoch 0이 10개, Epoch 1이 10개...)
combined_df = pd.concat(all_dfs)

# --- 4. 'epoch'별로 그룹화하여 모든 지표의 평균 계산 ---
# 'epoch'을 기준으로 그룹화하고, 나머지 모든 컬럼의 평균을 냅니다.
average_df = combined_df.groupby('epoch').mean()

# --- 5. 평균 데이터를 새 CSV 파일로 저장 ---
output_csv_path = os.path.join(OUTPUT_DIR, f'avg_results_{PROJECT_DIR}.csv')
average_df.to_csv(output_csv_path)
print(f"💾 평균 데이터 CSV가 '{output_csv_path}'에 저장되었습니다.")

# --- 6. "평균 학습 그래프" 생성 및 저장 ---
print("📊 평균 학습 그래프를 생성 중입니다...")

# 폰트 깨짐 방지 (필요시 주석 해제)
# plt.rcParams['font.family'] = 'Malgun Gothic' # Windows
# plt.rcParams['axes.unicode_minus'] = False 

# --- 그래프 1: Box Loss (train vs val) ---
plt.figure(figsize=(10, 6))
plt.plot(average_df.index, average_df['train/box_loss'], label='Average Train Box Loss')
plt.plot(average_df.index, average_df['val/box_loss'], label='Average Val Box Loss')
plt.title(f'Average Box Loss (K={K} Fold) - {PROJECT_DIR}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, f'avg_box_loss_{PROJECT_DIR}.png'))
plt.close()

# --- 그래프 2: Class Loss (train vs val) ---
plt.figure(figsize=(10, 6))
plt.plot(average_df.index, average_df['train/cls_loss'], label='Average Train Class Loss')
plt.plot(average_df.index, average_df['val/cls_loss'], label='Average Val Class Loss')
plt.title(f'Average Class Loss (K={K} Fold) - {PROJECT_DIR}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, f'avg_cls_loss_{PROJECT_DIR}.png'))
plt.close()

# --- 그래프 3: mAP (val) ---
plt.figure(figsize=(10, 6))
plt.plot(average_df.index, average_df['metrics/mAP50(B)'], label='Average mAP@0.5')
plt.plot(average_df.index, average_df['metrics/mAP50-95(B)'], label='Average mAP@0.5:0.95')
plt.title(f'Average Validation mAP (K={K} Fold) - {PROJECT_DIR}')
plt.xlabel('Epoch')
plt.ylabel('mAP Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, f'avg_mAP_{PROJECT_DIR}.png'))
plt.close()

# --- 그래프 4: Precision & Recall (val) ---
plt.figure(figsize=(10, 6))
plt.plot(average_df.index, average_df['metrics/precision(B)'], label='Average Precision')
plt.plot(average_df.index, average_df['metrics/recall(B)'], label='Average Recall')
plt.title(f'Average Validation P & R (K={K} Fold) - {PROJECT_DIR}')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, f'avg_PR_{PROJECT_DIR}.png'))
plt.close()

print(f"✅ 평균 그래프 4개가 '{OUTPUT_DIR}'에 저장되었습니다.")
print("🎉 K-Fold 평균 분석 완료!")