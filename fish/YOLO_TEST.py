from ultralytics import YOLO
import numpy as np

# 1. 모델 로드
model = YOLO('50per_re/weights/best.pt')
model.to('cpu')

# 2. 평가 실행
metrics = model.val(
    data='added_fish_re/50per/data.yaml',
    split='test',
    imgsz=640,
    batch=16,
    save=True,
    project='YOLO_result',
    name='fish',
    exist_ok=True
)

# 3. 주요 지표 출력
print("\n[YOLO 평가 지표 요약]")
print(f"mAP@0.5:          {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:     {metrics.box.map:.4f}")
print(f"Precision (mean): {metrics.box.mp:.4f}")
print(f"Recall (mean):    {metrics.box.mr:.4f}")

# 4. TP, FP, FN 계산 (클래스별)
print("\n[클래스별 TP / FP / FN 계산]")
for i, name in enumerate(metrics.names):
    p = metrics.box.p[i]       # precision
    r = metrics.box.r[i]       # recall
    ap = metrics.box.ap[i]     # average precision
    f1 = metrics.box.f1[i]     # f1 score

    # 추정: FN = TP*(1/Recall - 1), FP = TP*(1/Precision - 1)
    tp = metrics.box.tp[i] if hasattr(metrics.box, 'tp') else np.nan  # YOLOv8에서는 기본 없음
    fn = np.nan
    fp = np.nan
    if r > 0:
        tp = 1000  # 임의 기준. 실제로는 TP, FP, FN 수치를 수동으로 파악해야 정확
        fn = tp * (1/r - 1)
    if p > 0:
        fp = tp * (1/p - 1)

    print(f"- {name}: Precision={p:.3f}, Recall={r:.3f}, AP@50={ap:.3f}, F1={f1:.3f}")
    if not np.isnan(fp):
        print(f"    (추정) TP={tp:.0f}, FP={fp:.0f}, FN={fn:.0f}")

print("\n🔍 confusion_matrix.png 확인: YOLO_result/origin/ 또는 runs/val/ 경로 안에 생성됨")