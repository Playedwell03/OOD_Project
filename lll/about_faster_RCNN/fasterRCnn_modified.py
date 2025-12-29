# ==========================================================
# Faster R-CNN 모델을 5개 데이터셋에 대해 자동 학습하는 스크립트
# - 각 데이터셋은 COCO format 사용
# - Faster R-CNN (ResNet50 + FPN) 기반 (Pretrained X, Scratch O)
# - epoch마다 train/val loss 기록 및 best model 저장
# ==========================================================

import os, time, json, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor
from pycocotools.coco import COCO
import cv2

# ==========================================================
# [1] 사용자 환경에서 반드시 수정해야 하는 부분
# (train_root, train_json, val_root, val_json, run_name)
# ==========================================================
DATASETS = [
    ("A_tvt/train", "A_tvt/train/train.json",
     "A_tvt/valid",   "A_tvt/valid/valid.json",   "A_tvt"),
     ("ood_10per_cropped/train", "ood_10per_cropped/train/train.json",
     "ood_10per_cropped/valid",   "ood_10per_cropped/valid/valid.json",   "ood_10per_cropped"),
     ("ood_30per_cropped/train", "ood_30per_cropped/train/train.json",
     "ood_30per_cropped/valid",   "ood_30per_cropped/valid/valid.json",   "ood_30per_cropped"),
     ("ood_50per_cropped/train", "ood_50per_cropped/train/train.json",
     "ood_50per_cropped/valid",   "ood_50per_cropped/valid/valid.json",   "ood_50per_cropped"),
]

# ==========================================================
# [2] 학습 하이퍼파라미터
# ==========================================================
NUM_CLASSES = 17 
EPOCHS = 50
BATCH_SIZE = 4
LR = 0.005
MOMENTUM = 0.9
WD = 0.0005
STEP_SIZE = 30
GAMMA = 0.1
NUM_WORKERS = 4  # 문제 발생 시 0으로 수정
OUTDIR = "models_faster_RCNN_Scratch_test_batch4" # 출력 폴더명 변경 (구분 위해)
SEED = 42

# ----------------------------------------------------------
# 랜덤 시드 고정 (재현성 확보)
# ----------------------------------------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ==========================================================
# [3] COCO Dataset Loader
# (이미지 + bbox + label 불러오기)
# ==========================================================
class CocoDet(torch.utils.data.Dataset):
    def __init__(self, split_root, ann_file):
        """
        split_root: 예) ".../train" (아래에 images/ 폴더가 있어야 함)
        ann_file:   예) ".../train.json" (COCO 형식)
        """
        self.split_root = split_root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.getImgIds()))

        # category_id를 1..K 범위의 연속 라벨로 매핑 (0은 background)
        self.cat2contig = {cid: i for i, cid in enumerate(sorted(self.coco.getCatIds()))}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.coco.loadImgs(img_id)[0]

        # JSON의 file_name은 images/ 기준 상대경로
        img_path = os.path.join(self.split_root, "images", info["file_name"])
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(img_path)

        # BGR → RGB 변환 후 tensor 변환
        # (주의: Scratch 학습 시에는 정규화(Normalize)가 더 중요할 수 있으나,
        #  Faster R-CNN 내부에 기본 transform이 포함되어 있어 to_tensor만 해도 작동함)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = to_tensor(img)  # (C,H,W) float32 [0,1]

        # annotation 로드
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []

        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue

            # COCO bbox → xyxy 형식
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat2contig[a["category_id"]] + 1)  # label 1부터 시작
            areas.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }
        return img, target


# ----------------------------------------------------------
# DataLoader가 detection용 batch 묶을 때 필요
# ----------------------------------------------------------
def collate_fn(batch):
    return tuple(zip(*batch))


# ==========================================================
# [4] 빠른 간이 검증 (Faster R-CNN은 train 모드에서만 loss 출력)
# ==========================================================
@torch.no_grad()
def quick_val_loss(model, loader, device):
    """
    Faster R-CNN은 eval()에서는 loss를 반환하지 않음.
    따라서 train() 상태에서 targets를 넣어 loss만 계산.
    """
    was_training = model.training
    model.train(True)

    total, n = 0.0, 0
    for imgs, targets in loader:
        imgs = [i.to(device) for i in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())
        total += loss.item()
        n += 1

        if n >= 50:  # 너무 오래 돌지 않게 제한
            break

    model.train(was_training)
    return total / max(n, 1)


# ==========================================================
# [5] 하나의 데이터셋을 학습시키는 함수
# ==========================================================
def train_one_dataset(run_name, train_root, train_json, val_root, val_json, device):
    os.makedirs(os.path.join(OUTDIR, run_name), exist_ok=True)
    log_path = os.path.join(OUTDIR, run_name, "log.jsonl")

    # Dataset 생성
    train_ds = CocoDet(train_root, train_json)
    val_ds   = CocoDet(val_root, val_json)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # ------------------------------------------------------
    # Faster R-CNN 모델 생성 (ResNet50 + FPN) - Scratch Mode
    # weights=None: COCO 사전학습 가중치 사용 안 함
    # weights_backbone=None: ImageNet 백본 가중치 사용 안 함
    # ------------------------------------------------------
    print(f"[{run_name}] Building model from scratch (No Pretraining)...")
    model = fasterrcnn_resnet50_fpn(
        num_classes=NUM_CLASSES,
        weights=None,          # Detector 가중치 초기화
        weights_backbone=None  # Backbone(ResNet) 가중치 초기화
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WD)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = float("inf")
    last_ckpt = os.path.join(OUTDIR, run_name, "last.pth")
    best_ckpt = os.path.join(OUTDIR, run_name, "best.pth")

    # 로그 파일 초기화
    with open(log_path, "w", encoding="utf-8") as lf:
        pass

    # ======================================================
    # 학습 루프
    # ======================================================
    for epoch in range(1, EPOCHS + 1):
        model.train(True)
        epoch_loss = 0.0

        for imgs, targets in train_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # AMP 가속
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()

        sch.step()
        train_loss = epoch_loss / max(1, len(train_loader))

        # 빠른 검증 loss
        val_loss = quick_val_loss(model, val_loader, device)

        # 로그 저장
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "time": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{run_name}] Epoch {epoch}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")

        # 체크포인트 저장
        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_val}, best_ckpt)

    print(f"[{run_name}] done. best_val={best_val:.4f}  -> saved to {best_ckpt}")


# ==========================================================
# [6] 전체 데이터셋 순차 학습
# ==========================================================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    for train_root, train_json, val_root, val_json, name in DATASETS:
        # 경로 검사 (오타 즉시 체크됨)
        assert os.path.isdir(train_root) and os.path.isfile(train_json), \
            f"bad train paths: {train_root}, {train_json}"
        assert os.path.isdir(val_root) and os.path.isfile(val_json), \
            f"bad val paths: {val_root}, {val_json}"

        train_one_dataset(name, train_root, train_json, val_root, val_json, device)


if __name__ == "__main__":
    main()
    