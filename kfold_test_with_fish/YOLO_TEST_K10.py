from ultralytics import YOLO
import os

# --- 1. 기본 설정 ---
K = 10  # K-Fold 횟수

# 10개의 '학습된 모델'이 있는 폴더
MODELS_BASE_DIR = 'models_k10_runs_70per' 
# 10개의 'OOD가 포함된 데이터셋'이 있는 폴더
DATA_BASE_DIR = 'A_k10_runs_70per'  

# --- 2. 평가/예측 결과 저장 위치 설정 ---

# 1. model.val()의 시각화 결과(PR curve 등)가 저장될 프로젝트
EVAL_PROJECT = 'result_k_10_70per'
# 2. model.predict()의 예측 이미지가 저장될 프로젝트
PREDICT_PROJECT = 'predict_imgs_k_10_70per'
# 3. 10개 Run의 'metrics.txt' 요약 파일이 저장될 폴더
METRICS_SUMMARY_DIR = 'k10_metrics_summary_70per' 

os.makedirs(METRICS_SUMMARY_DIR, exist_ok=True)
all_metrics_list = [] # 10개 Run의 지표를 저장할 리스트

print(f"--- K={K} Fold Evaluation & Prediction Start ---")
print(f"Data source: {DATA_BASE_DIR}")
print(f"Models source: {MODELS_BASE_DIR}")

# --- 3. K=10 (1~10) 루프 실행 ---
for i in range(1, K + 1):
    run_name = f'run_{i}_test'
    
    print("\n" + "="*60)
    print(f"🚀 [Run {i}/{K}] STARTING: {run_name}")
    print("="*60 + "\n")

    # --- 4. 이 Run에 필요한 경로 정의 ---
    model_path = os.path.join(MODELS_BASE_DIR, run_name, 'weights', 'best.pt')
    data_yaml_path = os.path.join(DATA_BASE_DIR, run_name, 'data.yaml')
    predict_source_path = os.path.join(DATA_BASE_DIR, run_name, 'test', 'images')

    # --- 5. 파일 존재 여부 확인 ---
    if not os.path.exists(model_path):
        print(f"⚠️ 'best.pt' 모델을 찾을 수 없습니다: {model_path}. 이 Run은 건너뜁니다.")
        continue
    if not os.path.exists(data_yaml_path):
        print(f"⚠️ 'data.yaml' 파일을 찾을 수 없습니다: {data_yaml_path}. 이 Run은 건너뜁니다.")
        continue
    if not os.path.exists(predict_source_path):
        print(f"⚠️ 'test/images' 폴더를 찾을 수 없습니다: {predict_source_path}. 이 Run은 건너뜁니다.")
        continue
        
    # --- 6. 모델 로드 ---
    try:
        model = YOLO(model_path)
        model.to('cpu') # 요청하신 대로 CPU로 설정
        print(f"✅ Model loaded: {model_path}")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}. 이 Run은 건너뜁니다.")
        continue

    #####################
    # 7. 성능 평가 수행 #
    #####################
    print(f"  [1/2] Evaluating 'test' split...")
    try:
        metrics = model.val(
            data=data_yaml_path,
            split='test',        # ✅ 'test' 스플릿을 사용하도록 명시
            imgsz=416,
            batch=16,
            project=EVAL_PROJECT,
            name=run_name,
            exist_ok=True,
            device='cpu'         # CPU에서 평가 수행
        )

        # 평가 지표 추출
        map50 = metrics.box.map50
        map_all = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr
        
        # 최종 요약을 위해 리스트에 추가
        all_metrics_list.append({
            'run': run_name,
            'map50': map50,
            'map': map_all,
            'precision': precision,
            'recall': recall
        })

        # 평가 지표 출력
        print(f"    📊 [Metrics for {run_name}]")
        print(f"    mAP@0.5:          {map50:.4f}")
        print(f"    mAP@0.5:0.95:     {map_all:.4f}")

        # 개별 평가 지표 파일 저장 (요약 폴더에)
        output_dir = os.path.join(METRICS_SUMMARY_DIR, run_name)
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, 'metrics.txt')

        with open(metrics_path, 'w') as f:
            f.write(f"[모델 평가 지표: {run_name}]\n")
            f.write(f"mAP@0.5:          {map50:.4f}\n")
            f.write(f"mAP@0.5:0.95:     {map_all:.4f}\n")
            f.write(f"Precision (mean): {precision:.4f}\n")
            f.write(f"Recall (mean):    {recall:.4f}\n")
        
        print(f"    -> 개별 metrics.txt 저장 완료: {metrics_path}")

    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}. 예측 단계로 넘어갑니다.")

    #######################################
    # 8. 예측 결과 이미지 저장 (시각화용) #
    #######################################
    print(f"  [2/2] Predicting on 'test/images'...")
    try:
        results = model.predict(
            source=predict_source_path,
            imgsz=416,
            save=True,
            project=PREDICT_PROJECT,
            name=run_name,
            exist_ok=True,
            batch=16,
            device='cpu'
        )
        print(f"    -> 예측 이미지 저장 완료 (Project: {PREDICT_PROJECT})")
    
    except Exception as e:
        print(f"❌ 예측 중 오류 발생: {e}")
        
    print(f"✅ [Run {i}/{K}] FINISHED: {run_name}")


# --- 9. [최종 요약] 10개 Run의 평균 mAP 계산 ---
print("\n" + "="*60)
print("🎉 All 10 K-Fold evaluations are complete.")
print(f"📊 [K-Fold 최종 요약 (from {DATA_BASE_DIR})]")
print("="*60 + "\n")

if not all_metrics_list:
    print("❌ 계산된 metrics가 없습니다. 요약을 건너뜁니다.")
else:
    # mAP@0.5 기준으로 정렬하여 출력
    all_metrics_list.sort(key=lambda x: x['map50'], reverse=True)
    
    summary_file_path = os.path.join(METRICS_SUMMARY_DIR, '_K10_Final_Summary.txt')
    
    with open(summary_file_path, 'w') as f:
        f.write(f"[K-Fold 최종 요약: {DATA_BASE_DIR}]\n\n")
        f.write("--- 개별 Run 성능 (mAP@0.5 기준 정렬) ---\n")
        
        for metrics in all_metrics_list:
            line = f"  {metrics['run']}: mAP@0.5={metrics['map50']:.4f}, mAP@.5:.95={metrics['map']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}\n"
            print(line, end='')
            f.write(line)

        # 평균 계산
        avg_map50 = sum(m['map50'] for m in all_metrics_list) / len(all_metrics_list)
        avg_map = sum(m['map'] for m in all_metrics_list) / len(all_metrics_list)
        avg_p = sum(m['precision'] for m in all_metrics_list) / len(all_metrics_list)
        avg_r = sum(m['recall'] for m in all_metrics_list) / len(all_metrics_list)
        
        print("-------------------------------------------------")
        print(f"  🔥 AVERAGE mAP@0.5:      {avg_map50:.4f}")
        print(f"  🔥 AVERAGE mAP@.5:.95: {avg_map:.4f}")
        print(f"  🔥 AVERAGE Precision:  {avg_p:.4f}")
        print(f"  🔥 AVERAGE Recall:     {avg_r:.4f}")

        f.write("\n--- K-Fold 평균 (N=10) ---\n")
        f.write(f"AVERAGE mAP@0.5:      {avg_map50:.4f}\n")
        f.write(f"AVERAGE mAP@.5:.95: {avg_map:.4f}\n")
        f.write(f"AVERAGE Precision:  {avg_p:.4f}\n")
        f.write(f"AVERAGE Recall:     {avg_r:.4f}\n")
        
    print(f"\n✅ K-Fold 최종 요약 파일 저장 완료!\n   -> {summary_file_path}")