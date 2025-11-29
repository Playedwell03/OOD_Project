import os
import shutil
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import classification_report

# 클래스 정의
CLASS_NAMES = ['crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn']

CLASS_COUNT = len(CLASS_NAMES)
CLASS_MAP = {str(i): name for i, name in enumerate(CLASS_NAMES)}

# YOLO -> 분류용 디렉토리 변환 함수
def convert_yolo_dataset_to_classification(input_dir, output_dir):
    images_dir = os.path.join(input_dir, 'images')
    labels_dir = os.path.join(input_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)

    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            continue

        class_id = lines[0].split()[0]
        class_name = CLASS_MAP.get(class_id)
        if class_name is None:
            continue

        img_filename = label_file.replace('.txt', '.jpg')
        img_src_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(img_src_path):
            continue

        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        shutil.copy(img_src_path, os.path.join(class_folder, img_filename))

# 데이터 변환
convert_yolo_dataset_to_classification('v1_1_plus_v1_2_for_learn/train', 'v1_1_plus_v1_2_for_learn_cls/train')
convert_yolo_dataset_to_classification('v1_1_plus_v1_2_for_learn/valid', 'v1_1_plus_v1_2_for_learn_cls/val')

# Transform 설정
transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 데이터셋 로드
image_datasets = {
    'train': datasets.ImageFolder('v1_1_plus_v1_2_for_learn_cls/train', transform=transform['train']),
    'val': datasets.ImageFolder('v1_1_plus_v1_2_for_learn_cls/val', transform=transform['val'])
}
data_loaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=32, shuffle=False)
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 정의
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, CLASS_COUNT)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
epochs = 10
train_acc_history, val_acc_history = [], []

for epoch in range(epochs):
    model.train()
    correct, total = 0, 0
    for inputs, labels in data_loaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    train_acc = correct / total
    train_acc_history.append(train_acc)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

# 모델 저장
os.makedirs('model/v1_1_plus_v1_2_for_learn', exist_ok=True)
torch.save(model.state_dict(), 'model/v1_1_plus_v1_2_for_learn/resnet50_model.pth')

# 학습 그래프 저장
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='train_acc')
plt.plot(val_acc_history, label='val_acc')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model/v1_1_plus_v1_2_for_learn/train_accuracy.png')
plt.close()

# 클래스별 정밀도, 재현율, F1
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in data_loaders['val']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
print(report)
with open('model/v1_1_plus_v1_2/classification_report.txt', 'w') as f:
    f.write(report)