import torch
from pathlib import Path
import sys
import yolov5.val as val
val_run = val.run

# 1. 평가 실행
results, maps, times = val_run(
    weights='YOLO_cleaned_model/weights/best.pt',
    data='final_data_trash_removed_for_learn/data.yaml',
    imgsz=640,
    batch_size=16,
    task='test',        # test split 사용
    save_json=False,
    save_hybrid=False,
    conf_thres=0.001,   # 낮은 threshold로 모든 detection 평가
    iou_thres=0.6,
    single_cls=False,
    augment=False,
    verbose=False,
    save_txt=False,
    save_conf=False,
    save_crop=False,
    nosave=True,
    exist_ok=True,
    name='custom_eval'
)

# 2. 클래스 이름 가져오기
from utils.general import check_yaml
from utils.datasets import LoadImagesAndLabels

data_dict = check_yaml('final_data_trash_removed_for_learn/data.yaml')
names = data_dict['names']

# 3. 결과 출력
print(f"\n[📊 YOLOv5 평가 지표 요약]")
print(f"mAP@0.5:      {results[0]:.4f}")
print(f"mAP@0.5:0.95: {results[1]:.4f}")
print(f"Precision:    {results[2]:.4f}")
print(f"Recall:       {results[3]:.4f}")
print(f"F1-score:     {results[4]:.4f}")

# 4. 클래스별 AP, Precision, Recall
print(f"\n[📈 클래스별 평가 지표]")
for i, name in enumerate(names):
    ap = maps[i] if maps[i] is not None else 0.0
    # F1은 계산되지 않으므로 Precision/Recall 기반으로 수동 계산 가능
    precision = results[5][i] if i < len(results[5]) else 0.0
    recall = results[6][i] if i < len(results[6]) else 0.0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    print(f"- {name}: AP@0.5={ap:.3f}, Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")