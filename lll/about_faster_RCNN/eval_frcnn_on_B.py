import os
import json
import numpy as np
import torch
import cv2

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ==========================================================
# numpy 구버전 호환 처리 (COCOeval 사용 시 Warning 방지 목적)
# ==========================================================
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "object"):
    np.object = object


# ==========================================================
# 평가할 B 데이터셋 경로
# ==========================================================
B_ROOT = "A_tvt/test"
B_JSON = "A_tvt/test/test.json"

OUTDIR = "eval_faster_RCNN_test_batch4"


# ==========================================================
# 1) Faster R-CNN 모델 체크포인트 경로
# ==========================================================
FRCNN_CKPTS = {
    "A_tvt":   "models_faster_RCNN_Scratch_test_batch4/A_tvt/best.pth",
    "10per_cropped":       "models_faster_RCNN_Scratch_test_batch4/ood_10per_cropped/best.pth",
    "30per_cropped":     "models_faster_RCNN_Scratch_test_batch4/ood_30per_cropped/best.pth",
    "50per_cropped":   "models_faster_RCNN_Scratch_test_batch4/ood_50per_cropped/best.pth",
}

NUM_CLASSES_FRCNN = 17  # Faster R-CNN 학습 시 사용한 클래스 수 그대로


# ==========================================================
# 공용: COCO 결과 저장 + COCOeval 실행
# ==========================================================
def save_per_class_ap(coco_gt, coco_dt, save_csv, cat_ids=None):
    """클래스별 AP 계산 후 CSV 저장"""
    import csv

    if cat_ids is None:
        cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    cat_id_to_name = {c["id"]: c["name"] for c in cats}

    rows = []
    for cid in cat_ids:
        ce = COCOeval(coco_gt, coco_dt, "bbox")
        ce.params.catIds = [cid]  # ← 클래스별 평가
        ce.evaluate()
        ce.accumulate()
        ce.summarize()

        rows.append({
            "category_id": cid,
            "category_name": cat_id_to_name.get(cid, str(cid)),
            "AP@[.50:.95]": float(ce.stats[0]),
            "AP@0.50": float(ce.stats[1]),
            "AP@0.75": float(ce.stats[2]),
        })

    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    with open(save_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def coco_eval_from_results(coco_gt, results, out_json, per_class_csv=None, restrict_cat_ids=None):
    """COCO detection results → mAP 계산"""
    if len(results) == 0:
        print(f"⚠ detection 결과가 비어있음 → mAP=0 처리")
        return {"AP@[.50:.95]": 0.0, "AP@0.50": 0.0, "AP@0.75": 0.0}

    # ==========================================================
    # [수정됨] COCO 객체에 'info' 키가 없으면 빈 딕셔너리 추가하여 에러 방지
    # ==========================================================
    if "info" not in coco_gt.dataset:
        coco_gt.dataset["info"] = {}
    # ==========================================================

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(out_json)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    if restrict_cat_ids is not None:
        coco_eval.params.catIds = restrict_cat_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 클래스별 CSV 저장
    if per_class_csv is not None:
        try:
            save_per_class_ap(coco_gt, coco_dt, per_class_csv, cat_ids=restrict_cat_ids)
            print(f"  → per-class AP 저장됨: {per_class_csv}")
        except Exception as e:
            print(f"  ⚠ per-class AP 저장 실패: {e}")

    stats = {
        "AP@[.50:.95]": float(coco_eval.stats[0]),
        "AP@0.50": float(coco_eval.stats[1]),
        "AP@0.75": float(coco_eval.stats[2]),
    }
    return stats


# ==========================================================
# Faster R-CNN 평가용 Dataset
# ==========================================================
class CocoDetEval(torch.utils.data.Dataset):
    """COCO 평가용 이미지 + ID만 반환"""
    def __init__(self, root, ann_file):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.getImgIds()))
        self.cats = sorted(self.coco.getCatIds())

        # 연속 라벨 ↔ COCO category_id 매핑
        self.cat2contig = {cid: i for i, cid in enumerate(self.cats)}
        self.contig2cat = {i: cid for i, cid in enumerate(self.cats)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]
        rel = info["file_name"]

        # 이미지 경로 두 가지 케이스 대응
        cand1 = os.path.join(self.root, "images", rel)
        cand2 = os.path.join(self.root, rel)
        img_path = cand1 if os.path.isfile(cand1) else cand2

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        return to_tensor(img), img_id


def collate_eval(batch):
    imgs, ids = zip(*batch)
    return list(imgs), list(ids)


# ==========================================================
# Faster R-CNN 모델들을 B_all에 대해 평가
# ==========================================================
def eval_frcnn_on_B():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = CocoDetEval(B_ROOT, B_JSON)
    loader = DataLoader(dataset, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate_eval)

    coco_gt = dataset.coco
    summary = {}

    for run_name, ckpt_path in FRCNN_CKPTS.items():
        if not os.path.isfile(ckpt_path):
            print(f"⚠ 모델 없음: {ckpt_path} → 스킵")
            continue

        print(f"\n=== [Faster R-CNN / {run_name}] 평가 시작 ===")
        model = fasterrcnn_resnet50_fpn(num_classes=NUM_CLASSES_FRCNN)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        results = []
        with torch.no_grad():
            for imgs, img_ids in loader:
                outs = model([img.to(device) for img in imgs])
                for out, img_id in zip(outs, img_ids):
                    boxes = out["boxes"].cpu().numpy()
                    scores = out["scores"].cpu().numpy()
                    labels = out["labels"].cpu().numpy()

                    keep = scores >= 0.001
                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                    if len(boxes) == 0:
                        continue

                    wh = boxes[:, 2:4] - boxes[:, 0:2]
                    xywh = np.concatenate([boxes[:, :2], wh], axis=1)

                    for i in range(len(xywh)):
                        contig = int(labels[i] - 1)
                        if contig not in dataset.contig2cat:
                            continue
                        cat_id = dataset.contig2cat[contig]

                        results.append({
                            "image_id": int(img_id),
                            "category_id": int(cat_id),
                            "bbox": xywh[i].tolist(),
                            "score": float(scores[i]),
                        })

        out_json = os.path.join(OUTDIR, "frcnn", f"{run_name}_pred_B.json")
        csv_path = out_json.replace(".json", "_per_class.csv")

        stats = coco_eval_from_results(
            coco_gt, results, out_json,
            per_class_csv=csv_path,
            restrict_cat_ids=coco_gt.getCatIds()
        )

        summary[run_name] = stats

    return summary


# ==========================================================
# main
# ==========================================================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("========== Faster R-CNN 모델들 B_all mAP 평가 ==========\n")

    # 1) Faster R-CNN 평가 실행
    frcnn_summary = eval_frcnn_on_B()

    # 2) YOLO는 비활성화됨
    yolo_summary = {}

    # 3) 통합 결과 저장
    combined = {
        "faster_rcnn": frcnn_summary,
        "yolo": yolo_summary,
    }
    sum_path = os.path.join(OUTDIR, "summary_B_all.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\n종합 summary 저장 완료: {sum_path}\n")

    print("=== Faster R-CNN (AP@[.50:.95] / AP50 / AP75) ===")
    for name, s in frcnn_summary.items():
        print(f"{name:22s} {s['AP@[.50:.95]']:.4f} / {s['AP@0.50']:.4f} / {s['AP@0.75']:.4f}")


if __name__ == "__main__":
    main()