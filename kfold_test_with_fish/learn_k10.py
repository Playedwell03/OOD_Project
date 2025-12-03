import os
from ultralytics import YOLO

# --- 1. 기본 설정 ---
K = 10
BASE_DATA_DIR = 'A_k10_runs_70per'    # 10개의 데이터셋 폴더가 있는 곳
PROJECT_NAME = 'models_k10_runs_70per'  # 모든 학습 결과(10개)가 저장될 부모 폴더
MODEL_CONFIG = 'yolov5s.yaml'   # 가중치 없는 YOLOv5s 구조

print(f"--- K={K} Fold Training Start ---")
print(f"Results will be saved to: {PROJECT_NAME}")

# --- 2. K=10 (1~10) 루프 실행 ---
for i in range(1, K + 1):
    run_name = f'run_{i}_test'
    data_yaml_path = os.path.join(BASE_DATA_DIR, run_name, 'data.yaml')

    # 현재 작업중인 폴더가 있는지 확인
    if not os.path.exists(data_yaml_path):
        print(f"⚠️ WARNING: {data_yaml_path} 를 찾을 수 없습니다. 이 Run은 건너뜁니다.")
        continue

    print("\n" + "="*60)
    print(f"🚀 [Run {i}/{K}] STARTING TRAINING: {run_name}")
    print(f"     Data YAML: {data_yaml_path}")
    print("="*60 + "\n")

    # --- 3. 모델 로드 (★매우 중요★) ---
    # 루프 안에서 매번 모델을 새로 로드해야 합니다.
    # 이렇게 해야 10개의 모델이 "가중치 랜덤 초기화" 상태에서
    # (yolov5s.yaml) 개별적으로 학습됩니다.
    model = YOLO(MODEL_CONFIG)

    # --- 4. 학습 실행 (파라미터 유지) ---
    try:
        model.train(
            data=data_yaml_path,
            epochs=50,
            imgsz=416,
            batch=16,
            device='0',          # ✅ GPU 0번
            name=run_name,       # ✅ 결과 폴더 이름 (예: run_1_test, run_2_test)
            project=PROJECT_NAME,# ✅ 모든 Run이 이 폴더 하위에 저장됨
            patience=0,
            optimizer='SGD',
            workers=8,
            exist_ok=True  # 이미 폴더가 있어도 덮어쓰며 진행
        )
        print(f"\n✅ [Run {i}/{K}] FINISHED TRAINING: {run_name}")
    
    except Exception as e:
        print(f"\n❌ [Run {i}/{K}] FAILED TRAINING: {run_name} with error: {e}")
        print("   Skipping to the next run...")
        continue # 학습 중 오류가 발생해도 다음 K-Fold로 넘어갑니다.

print("\n" + "="*60)
print("🎉 All 10 K-Fold trainings are complete.")
print(f"Check results in '{PROJECT_NAME}' folder.")
print("="*60)