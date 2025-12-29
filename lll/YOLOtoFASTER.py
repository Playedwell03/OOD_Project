import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import glob

# ==========================================
# 1. Dataset 클래스 정의 (YOLO -> Faster R-CNN 변환기)
# ==========================================
class YoloToFasterRCNNDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms
        
        # 이미지 파일 리스트 (.jpg, .png 등) 가져오기
        # 대소문자 구분 없이 가져오려면 glob 패턴을 조정해야 하지만, 보통 jpg/png입니다.
        self.img_files = sorted(glob.glob(os.path.join(img_dir, '*.*')))
        
        # 이미지 확장자 필터링 (필요시)
        self.img_files = [f for f in self.img_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. 이미지 읽기
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        w_img, h_img = img.size # 픽셀 크기 (예: 640, 480)

        # 2. 라벨 파일 찾기 (.txt)
        # 이미지 파일명과 동일한 이름의 txt 파일을 찾습니다.
        label_file = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = list(map(float, line.strip().split()))
                
                # YOLO 포맷: class_id, center_x, center_y, width, height (0~1 정규화)
                class_id = int(parts[0])
                cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]

                # --- ✨ 좌표 변환 (핵심) ---
                # 1) 픽셀로 변환 (역정규화)
                cx *= w_img
                cy *= h_img
                w *= w_img
                h *= h_img

                # 2) (중심, 크기) -> (좌상단, 우하단) 변환
                x_min = cx - (w / 2)
                y_min = cy - (h / 2)
                x_max = cx + (w / 2)
                y_max = cy + (h / 2)

                boxes.append([x_min, y_min, x_max, y_max])
                
                # Faster R-CNN은 0번을 '배경'으로 쓰므로, 클래스 ID를 1씩 밀어줍니다.
                labels.append(class_id + 1) 

        # 3. 텐서 변환
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # 박스가 없는 경우 (배경 이미지)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # 면적
        target["iscrowd"] = torch.zeros((len(labels),), dtype=torch.int64) # 군집 여부 (기본 0)

        # 4. 이미지 텐서 변환 (transforms 적용)
        # 기본적으로 텐서 변환은 해줘야 합니다.
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            # transforms가 없어도 최소한 Tensor로는 바꿔야 모델에 들어갑니다.
            import torchvision.transforms as T
            img = T.ToTensor()(img)

        return img, target

# ==========================================
# 2. Faster R-CNN용 Collate Function
# ==========================================
def collate_fn(batch):
    return tuple(zip(*batch))

# ==========================================
# 3. 메인 실행 코드
# ==========================================
if __name__ == '__main__':
    # --- 경로 설정 (사용자님 폴더 구조에 맞게 수정하세요!) ---
    BASE_DIR = '50per' # 최상위 폴더
    
    # 각 폴더 경로 정의
    TRAIN_IMG = os.path.join(BASE_DIR, 'train/images')
    TRAIN_LBL = os.path.join(BASE_DIR, 'train/labels')

    VALID_IMG = os.path.join(BASE_DIR, 'valid/images') # 폴더명이 val 인지 valid 인지 확인!
    VALID_LBL = os.path.join(BASE_DIR, 'valid/labels')

    TEST_IMG  = os.path.join(BASE_DIR, 'test/images')
    TEST_LBL  = os.path.join(BASE_DIR, 'test/labels')

    BATCH_SIZE = 4

    print("🔄 데이터셋 로드 준비 중...")

    # 1. 데이터셋 인스턴스 생성
    train_dataset = YoloToFasterRCNNDataset(TRAIN_IMG, TRAIN_LBL)
    val_dataset   = YoloToFasterRCNNDataset(VALID_IMG, VALID_LBL)
    test_dataset  = YoloToFasterRCNNDataset(TEST_IMG, TEST_LBL)

    # 2. 데이터 로더 생성
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print("\n✅ 데이터 로드 완료!")
    print(f" - Train 데이터 수 : {len(train_dataset)}장")
    print(f" - Valid 데이터 수 : {len(val_dataset)}장")
    print(f" - Test  데이터 수 : {len(test_dataset)}장")

    # 3. 샘플 데이터 하나 뽑아보기 (검증)
    print("\n🔍 첫 번째 배치 데이터 검사 중...")
    try:
        # iter()로 첫 번째 배치를 가져옵니다.
        images, targets = next(iter(train_loader))
        
        print(f" - 배치 크기: {len(images)}")
        print(f" - 첫 번째 이미지 텐서 모양: {images[0].shape} (C, H, W)")
        print(f" - 첫 번째 이미지 라벨(Target): {targets[0]['labels']}")
        print(f" - 첫 번째 이미지 박스(Boxes): \n{targets[0]['boxes']}")
        
        print("\n🎉 성공! Faster R-CNN 학습 코드를 돌릴 준비가 되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생! 경로를 확인해주세요.\n에러 메시지: {e}")