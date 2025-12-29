import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt 

# ==============================================================================
# [ 1. 사용자 설정 (CONFIG) ] 
# ==============================================================================
CONFIG = {
    # 1. 파일 경로 설정
    "TRAIN_DATA_DIR": "50per/train",         # 학습용 데이터 경로
    "VALID_DATA_DIR": "50per/valid",         # 검증용 데이터 경로
    
    "OUTPUT_MODEL_PATH": "ood_50per_model.pth", # 최종 모델(Last) 저장명
    "OUTPUT_GRAPH_PATH": "50per_loss_graph.png", # 결과 그래프 저장명

    # 2. 데이터셋 및 모델 설정
    "NUM_CLASSES": 17,     # 클래스 개수 (★ 배경 포함! 예: 물체 16개 + 배경 1개 = 17)
    "IMG_SIZE": 416,       # 입력 이미지 크기
    
    # 3. 하이퍼 파라미터 (YOLO와 조건 통일: 50 Epochs)
    "BATCH_SIZE": 4,       
    "NUM_EPOCHS": 50,      
    "LEARNING_RATE": 0.005,
    "MOMENTUM": 0.9,       
    "WEIGHT_DECAY": 0.0005,
    
    # 4. 기타 설정
    "STEP_SIZE": 20,       # 학습률 감소 주기 (50에폭이니 20, 40에서 감소 추천)
    "GAMMA": 0.1,          # 학습률 감소 비율
    "NUM_WORKERS": 8       # Linux 환경 추천 설정
}

# ==============================================================================
# [ 2. 데이터셋 클래스 정의 (YOLO -> Faster R-CNN 변환 & 에러 방지 포함) ]
# ==============================================================================
class OODResearchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, width, height, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.width = width
        self.height = height
        
        # 이미지와 라벨 파일 리스트 정렬하여 로드
        self.imgs = list(sorted(os.listdir(os.path.join(root_dir, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root_dir, "labels"))))

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_path = os.path.join(self.root_dir, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((self.width, self.height)) # 리사이징
        img_tensor = transforms.ToTensor()(img)

        # 2. 라벨 로드 및 변환 (YOLO txt -> Faster R-CNN Box)
        label_path = os.path.join(self.root_dir, "labels", self.labels[idx])
        boxes = []
        labels_idx = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    # YOLO 포맷: class_id x_center y_center w h (0~1 Normalized)
                    parts = list(map(float, line.strip().split()))
                    class_id = int(parts[0])
                    x_c, y_c, w, h = parts[1], parts[2], parts[3], parts[4]

                    # (1) 절대 좌표(Pixel)로 변환
                    x_c *= self.width
                    y_c *= self.height
                    w *= self.width
                    h *= self.height

                    # (2) Center 좌표(x_c, y_c) -> Corner 좌표(x1, y1, x2, y2) 변환
                    x1 = x_c - w / 2
                    y1 = y_c - h / 2
                    x2 = x_c + w / 2
                    y2 = y_c + h / 2

                    # (3) [에러 방지] 이미지 밖으로 나가는 좌표 잘라내기 (Clamp)
                    x1 = max(0, min(x1, self.width))
                    y1 = max(0, min(y1, self.height))
                    x2 = max(0, min(x2, self.width))
                    y2 = max(0, min(y2, self.height))

                    # (4) [에러 방지] 유효한 박스(너비/높이가 0보다 큰 것)만 담기
                    if (x2 > x1) and (y2 > y1):
                        boxes.append([x1, y1, x2, y2])
                        labels_idx.append(class_id + 1) # 0은 배경이므로 클래스 ID + 1

        # 유효한 박스가 없거나 파일이 비어있는 경우 처리
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_idx = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels_idx = torch.as_tensor(labels_idx, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels_idx
        target["image_id"] = torch.tensor([idx])

        if self.transforms:
            img_tensor = self.transforms(img_tensor)

        return img_tensor, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

# [ 3. 학습 결과 그래프 저장 함수 ]
def plot_learning_curve(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(valid_losses, label='Validation Loss', color='red', linewidth=2, linestyle='--')
    plt.title('Training & Validation Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CONFIG["OUTPUT_GRAPH_PATH"], dpi=300)
    print(f"\n📊 그래프 저장 완료: {CONFIG['OUTPUT_GRAPH_PATH']}")

# ==============================================================================
# [ 4. 메인 실행 함수 (Best Model 저장 기능 추가됨) ]
# ==============================================================================
def main():
    # 4-1. 디바이스 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"🚀 Device Setting: {device}")
    
    # 4-2. 데이터셋 로드
    print("📂 데이터셋 로딩 중...")
    try:
        train_dataset = OODResearchDataset(CONFIG["TRAIN_DATA_DIR"], CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])
        valid_dataset = OODResearchDataset(CONFIG["VALID_DATA_DIR"], CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        print("경로를 다시 확인해주세요:", CONFIG["TRAIN_DATA_DIR"])
        return

    print(f"   ├─ Train 데이터: {len(train_dataset)}장")
    print(f"   └─ Valid 데이터: {len(valid_dataset)}장")

    # 4-3. 데이터 로더
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=CONFIG["NUM_WORKERS"], collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"], collate_fn=collate_fn)

    # 4-4. 모델 초기화
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, CONFIG["NUM_CLASSES"])
    model.to(device)

    # 4-5. 옵티마이저 & 스케줄러
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=CONFIG["LEARNING_RATE"], momentum=CONFIG["MOMENTUM"], weight_decay=CONFIG["WEIGHT_DECAY"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONFIG["STEP_SIZE"], gamma=CONFIG["GAMMA"])

    # --- Best Model 저장을 위한 변수 ---
    min_valid_loss = float('inf') 
    best_model_path = CONFIG["OUTPUT_MODEL_PATH"].replace(".pth", "_best.pth")
    
    train_loss_history = []
    valid_loss_history = []

    print(f"\n🔥 학습 시작... (총 {CONFIG['NUM_EPOCHS']} Epochs)")
    
    for epoch in range(CONFIG["NUM_EPOCHS"]):
        # [ TRAIN Loop ]
        model.train()
        train_epoch_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{CONFIG['NUM_EPOCHS']} [Train]")

        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets_list = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets_list)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_epoch_loss += losses.item()
            loop.set_postfix(loss=losses.item())
        
        avg_train_loss = train_epoch_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # [ VALID Loop ] - Train 모드 + No Grad로 Loss 계산
        with torch.no_grad():
            valid_epoch_loss = 0
            if len(valid_loader) > 0:
                for images, targets in valid_loader:
                    images = list(image.to(device) for image in images)
                    targets_list = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets_list)
                    losses = sum(loss for loss in loss_dict.values())
                    valid_epoch_loss += losses.item()
                avg_valid_loss = valid_epoch_loss / len(valid_loader)
            else:
                avg_valid_loss = 0.0

        valid_loss_history.append(avg_valid_loss)
        lr_scheduler.step()

        print(f"   >>> Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}")

        # [ Best Model 저장 ]
        if avg_valid_loss < min_valid_loss and avg_valid_loss > 0:
            min_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"   ⭐ Best Model 갱신! Saved to: {best_model_path}")

    # 최종 모델 저장 (Last)
    torch.save(model.state_dict(), CONFIG["OUTPUT_MODEL_PATH"])
    
    # 그래프 그리기
    plot_learning_curve(train_loss_history, valid_loss_history)
    
    print(f"\n🎉 학습 완료!")
    print(f"   1. 최종 모델(Last): {CONFIG['OUTPUT_MODEL_PATH']}")
    print(f"   2. 최고 성능 모델(Best): {best_model_path}  <-- OOD 테스트시 이거 추천")

if __name__ == "__main__":
    main()