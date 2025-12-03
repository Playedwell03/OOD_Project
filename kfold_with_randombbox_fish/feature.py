import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 학습된 모델 불러오기
model = torch.load('models_k10_runs_A/run_1_test/weights/best.pt')['model'].float()
model.eval()

# 2. 특징 벡터를 뽑기 위한 Hook 설정 (전자공학의 Probing)
# 보통 Detect 헤드 직전의 Backbone 끝단(마지막 Conv 레이어)을 찍습니다.
features = []
labels = []

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 1. 커스텀 데이터셋 클래스 정의 (전자공학의 'ADC' 역할: 아날로그 사진을 디지털 텐서로 변환)
class OODDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB") # 이미지 열기
        
        # 라벨은 t-SNE 그릴 때 색깔 구분을 위해 필요합니다.
        # 일단 OOD는 전부 1번 클래스, 정상은 0번 클래스 이런식으로 가칭을 붙여야 합니다.
        # 여기서는 파일명이나 폴더명으로 구분하거나, 일단 '0'(미정)으로 둡니다.
        label = 0 
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 2. 전처리 설정 (YOLO 입력 크기에 맞게 리사이징 중요!)
transform = transforms.Compose([
    transforms.Resize((640, 640)), # YOLOv5s 기본 입력 사이즈
    transforms.ToTensor(),         # 이미지를 0~1 사이의 텐서로 변환
])

# 3. 데이터로더 생성 (여기를 수정하세요!)
# OOD 이미지가 들어있는 폴더 경로를 넣어주세요.
img_dir = 'path/to/your/ood_images'  # <--- 실제 폴더 경로로 변경!!
dataset = OODDataset(img_dir, transform=transform)

# 배치 사이즈는 메모리 허용하는 만큼 (t-SNE용이라 1이어도 상관없음)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# --- 이제 아래부터 아까 드린 코드가 작동합니다 ---
# hook_fn 정의...
# model에 hook 걸기...
# for img, label in dataloader: ...

def hook_fn(module, input, output):
    # output이 특징 맵(Feature Map)입니다. 이걸 벡터로 평탄화(Flatten)해서 저장
    features.append(output.mean(dim=[2, 3]).detach().cpu().numpy()) 

# 모델의 마지막 레이어 찾아서 Hook 걸기 (예시)
# model.model[-2] 이런 식으로 Backbone 끝단을 찾습니다.
handle = model.model[-2].register_forward_hook(hook_fn)

# 3. 데이터 흘려보내기 (Inference)
# 가지고 계신 검증셋(val)이나 OOD 데이터를 모델에 통과시킵니다.
with torch.no_grad():
    for img, label in dataloader:
        model(img) # 모델 통과 -> Hook이 특징을 자동으로 낚아챔
        labels.extend(label)

# 4. 차원 축소 (512차원 -> 2차원)
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 5. 그리기
df = pd.DataFrame({'x': features_2d[:,0], 'y': features_2d[:,1], 'class': labels})
sns.scatterplot(data=df, x='x', y='y', hue='class')
plt.title("Feature Distribution (t-SNE)")
plt.show()