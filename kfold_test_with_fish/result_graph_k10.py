from ultralytics import YOLO
import os
import numpy as np                # 👈 [추가됨] 평균 계산을 위해 numpy 임포트
import matplotlib.pyplot as plt   # 👈 [추가됨] 그래프 생성을 위해 matplotlib 임포트

# --- 1. 기본 설정 ---
K = 10  # K-Fold 횟수

# 10개의 '학습된 모델'이 있는 폴더
MODELS_BASE_DIR = 'models_k10_runs_70per' 
# 10개의 'OOD가 포함된 데이터셋'이 있는 폴더
DATA_BASE_DIR = 'A_k10_runs_70per'  

# --- 2. 평가/예측 결과 저장 위치 설정 ---
EVAL_PROJECT = 'result_k_10_70per_2'
PREDICT_PROJECT = 'predict_imgs_k_10_70per_2'
# 3. 10개 Run의 요약 파일이 저장될 폴더
METRICS_SUMMARY_DIR = 'k10_result_metrics_70per_2' 

os.makedirs(METRICS_SUMMARY_DIR, exist_ok=True)
all_metrics_list = [] # 10개 Run의 '최종 스칼라 지표'를 저장할 리스트

# --- [추가됨] 평균 그래프를 위한 변수 설정 ---
# 10개의 '평가 커브' 데이터를 저장할 리스트
all_p_curves = []  # (Confidence, Precision)
all_r_curves = []  # (Confidence, Recall)
all_f1_curves = [] # (Confidence, F1)
all_pr_curves = [] # (Recall, Precision)

# 평균을 내기 위한 공통 x-축 (0부터 1까지 101개 지점)
# np.linspace(start, stop, num_points)
common_confidence_axis = np.linspace(0, 1, 101)
common_recall_axis = np.linspace(0, 1, 101)
# ---

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
    # 7. [수정됨] 성능 평가 수행 & 커브 데이터 추출 #
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

        # --- (A) 최종 스칼라(숫자) 지표 추출 ---
        map50 = metrics.box.map50
        map_all = metrics.box.map
        precision = metrics.box.mp
        recall = metrics.box.mr
        
        all_metrics_list.append({
            'run': run_name, 'map50': map50, 'map': map_all,
            'precision': precision, 'recall': recall
        })

        # 평가 지표 출력
        print(f"    📊 [Metrics for {run_name}]")
        print(f"    mAP@0.5:          {map50:.4f}")
        print(f"    mAP@0.5:0.95:     {map_all:.4f}")

        # --- [추가됨] (B) 커브 원시 데이터 추출 및 보간(Interpolation) ---
        print("    ... 평가 커브(P, R, F1, PR) 원시 데이터 추출 중 ...")
        
        # (중요!) metrics.curves가 None인지(감지된 객체가 0개인지) 확인
        if metrics.curves is None or 'Precision' not in metrics.curves:
            print("    ⚠️ 커브 데이터가 없습니다 (0 Detections?). 이 Run의 커브는 평균 계산에서 제외됩니다.")
        else:
            # 커브 데이터가 있을 때만 리스트에 추가
            p_data = metrics.curves['Precision']
            interp_p = np.interp(common_confidence_axis, p_data[0][::-1], p_data[1][::-1])
            all_p_curves.append(interp_p)

            r_data = metrics.curves['Recall']
            interp_r = np.interp(common_confidence_axis, r_data[0][::-1], r_data[1][::-1])
            all_r_curves.append(interp_r)

            f1_data = metrics.curves['F1']
            interp_f1 = np.interp(common_confidence_axis, f1_data[0][::-1], f1_data[1][::-1])
            all_f1_curves.append(interp_f1)
            
            pr_data = metrics.curves['PR']
            interp_pr = np.interp(common_recall_axis, pr_data[0], pr_data[1])
            all_pr_curves.append(interp_pr)
            print("    ... 커브 데이터 추출 성공.")

        # (기존) 개별 평가 지표 파일 저장
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
        print(f"❌ 평가 중 심각한 오류 발생: {e}. 이 Run은 건너뜁니다.")
        continue # 예외 발생 시 다음 Run으로

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


# --- 9. [최종 요약] 10개 Run의 평균 스칼라 지표 계산 ---
print("\n" + "="*60)
print("🎉 All 10 K-Fold evaluations are complete.")
print(f"📊 [K-Fold 최종 스칼라 요약 (from {DATA_BASE_DIR})]")
print("="*60 + "\n")

if not all_metrics_list:
    print("❌ 계산된 metrics가 없습니다. 스칼라 요약을 건너뜁니다.")
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

# --- 10. [추가됨] 최종 평균 평가 그래프 생성 ---
print("\n" + "="*60)
print(f"📊 [K-Fold 평균 평가 그래프 생성 (from {DATA_BASE_DIR})]")
print("="*60 + "\n")

# 커브 데이터가 하나라도 수집되었는지 확인
if not all_pr_curves:
    print("❌ 계산된 커브 데이터가 없습니다. 그래프 생성을 건너뜁니다.")
else:
    print(f"  ... {len(all_pr_curves)}개의 유효한 Run 데이터를 바탕으로 평균 그래프 생성 중 ...")
    
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # --- 그래프 1 (좌상단): Precision vs Confidence ---
    ax = axes[0, 0]
    avg_p_curve = np.mean(all_p_curves, axis=0) # 10개 커브 평균
    ax.plot(common_confidence_axis, avg_p_curve, label='Average P')
    ax.set_title(f'Average Precision vs Confidence Curve (N={len(all_p_curves)})')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)
    
    # --- 그래프 2 (우상단): Recall vs Confidence ---
    ax = axes[0, 1]
    avg_r_curve = np.mean(all_r_curves, axis=0) # 10개 커브 평균
    ax.plot(common_confidence_axis, avg_r_curve, label='Average R', color='orange')
    ax.set_title(f'Average Recall vs Confidence Curve (N={len(all_r_curves)})')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Recall')
    ax.legend()
    ax.grid(True)

    # --- 그래프 3 (좌하단): F1 vs Confidence ---
    ax = axes[1, 0]
    avg_f1_curve = np.mean(all_f1_curves, axis=0) # 10개 커브 평균
    ax.plot(common_confidence_axis, avg_f1_curve, label='Average F1', color='green')
    ax.set_title(f'Average F1 vs Confidence Curve (N={len(all_f1_curves)})')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True)

    # --- 그래프 4 (우하단): Precision vs Recall (PR Curve) ---
    ax = axes[1, 1]
    avg_pr_curve = np.mean(all_pr_curves, axis=0) # 10개 커브 평균
    ax.plot(common_recall_axis, avg_pr_curve, label='Average PR Curve', color='red')
    ax.set_title(f'Average Precision-Recall (PR) Curve (N={len(all_pr_curves)})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend()
    ax.grid(True)

    # --- 전체 저장 ---
    fig.suptitle(f'K={K} Fold Average Evaluation Curves - {DATA_BASE_DIR}', fontsize=24, y=1.03)
    plt.tight_layout()
    output_png_path = os.path.join(METRICS_SUMMARY_DIR, f'_K10_Final_Avg_Evaluation_Curves.png')
    
    # 
    # ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
    # 
    #           ✅✅✅ 이 부분이 수정되었습니다! ✅✅✅
    #           'dpi=300' (고해상도) 옵션을 제거했습니다.
    # 
    # ‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️‼️
    # 
    plt.savefig(output_png_path, bbox_inches='tight') 
    plt.close()

    print(f"✅ 2x2 평균 평가 그래프 1개가 '{output_png_path}'에 저장되었습니다.")

print("\n🎉 모든 작업이 완료되었습니다.")