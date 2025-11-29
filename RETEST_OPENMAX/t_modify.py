import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from scipy.stats import weibull_min
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil

# 설정
MODEL_PATH = 'output/resnet50_model.pth'
TEST_IMAGE_DIR = 'fish2/images'
TEST_LABEL_DIR = 'fish2/labels'
CLASS_NAMES = ['crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = len(CLASS_NAMES)
WEIBULL_TAIL_SIZE = 10
DIST_THRESHOLD = 45.0

# 모델 정의 및 로드
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Feature 추출기 정의
feature_extractor = nn.Sequential(*list(model.children())[:-1]).to(DEVICE)

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def fit_weibull(model, data_dir):
    class_features = {i: [] for i in range(NUM_CLASSES)}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feature = feature_extractor(image).squeeze().cpu().numpy()

            # ✅ feature도 정규화 (optional, 시각화나 Weibull 적합 정확도 향상 가능)
            feature = feature / np.linalg.norm(feature)
            class_features[class_idx].append(feature)

    weibull_models = {}
    mean_vecs = {}
    for class_idx, feats in class_features.items():
        feats = np.array(feats)

        # ✅ 평균 벡터 계산 후 정규화
        mean_vec = np.mean(feats, axis=0)
        mean_vec = mean_vec / np.linalg.norm(mean_vec)

        dists = pairwise_distances([mean_vec], feats, metric='euclidean').flatten()
        tail = np.sort(dists)[-WEIBULL_TAIL_SIZE:]
        shape, loc, scale = weibull_min.fit(tail, floc=0)
        weibull_models[class_idx] = (shape, loc, scale)
        mean_vecs[class_idx] = mean_vec

    return weibull_models, mean_vecs, class_features

from scipy.special import softmax as scipy_softmax

def compute_openmax_probability(logits, feature_vec, weibull_models, mean_vecs, alpha=5):
    logits = logits.cpu().numpy()
    feature_vec = feature_vec.cpu().numpy()
    feature_vec = feature_vec / np.linalg.norm(feature_vec)

    ranked = np.argsort(logits)[::-1]
    omega = np.ones(NUM_CLASSES)

    for i in range(min(alpha, NUM_CLASSES)):
        class_idx = ranked[i]
        mean_vec = mean_vecs[class_idx]
        mean_vec = mean_vec / np.linalg.norm(mean_vec)
        dist = np.linalg.norm(feature_vec - mean_vec)
        shape, loc, scale = weibull_models[class_idx]
        wscore = weibull_min.cdf(dist, shape, loc=loc, scale=scale)
        omega[class_idx] = np.clip(1 - wscore, 0.6, 1.0)

    modified_logits = logits * omega
    unknown_score = np.sum(logits * (1 - omega))
    openmax_logits = np.append(modified_logits, unknown_score)

    # 소프트맥스 다시 씌우기
    openmax_probs = scipy_softmax(openmax_logits)
    return openmax_probs

def plot_openmax_result(img_tensor, probs, img_name):
    os.makedirs('t2/plots', exist_ok=True)
    probs = np.array(probs)
    classes_with_unknown = CLASS_NAMES + ['unknown']

    img_np = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img_np = np.clip(img_np, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.imshow(img_np)
    plt.axis('off')
    plt.title(img_name)

    plt.subplot(2, 1, 2)
    bars = plt.bar(classes_with_unknown, probs)
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
    plt.ylabel("Confidence")
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(f't2/plots/{img_name.replace(".jpg", "")}_openmax.png')
    plt.close()

def run_openmax_inference(test_img_dir, test_label_dir, model, weibull_models, mean_vecs, img_output_path='t2/confusion_matrix.png', alpha=5):
    unknown_count = 0
    total = 0
    os.makedirs('t2/unknown', exist_ok=True)
    confusion_mat = np.zeros((NUM_CLASSES, NUM_CLASSES + 1), dtype=int)

    for img_name in os.listdir(test_img_dir):
        if not img_name.endswith('.jpg'):
            continue

        img_path = os.path.join(test_img_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(test_label_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            print(f"[경고] 라벨 없음: {label_path}")
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                continue
            try:
                true_label = int(lines[0].strip().split()[0])
            except:
                continue

        image_tensor = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feature = feature_extractor(image_tensor).squeeze()
            logits = model(image_tensor)

        openmax_probs = compute_openmax_probability(logits[0], feature, weibull_models, mean_vecs, alpha=alpha)
        pred = np.argmax(openmax_probs)
        total += 1

        plot_openmax_result(image_tensor, openmax_probs, img_name)

        feature_np = feature.cpu().numpy()
        dists = [np.linalg.norm(feature_np - mean_vecs[c]) for c in range(NUM_CLASSES)]
        min_dist = np.min(dists)

        if pred == NUM_CLASSES or min_dist > DIST_THRESHOLD:
            pred_label = NUM_CLASSES  # unknown
            unknown_count += 1
            shutil.copy(img_path, f't2/unknown/{img_name}')
        else:
            pred_label = pred

        if 0 <= true_label < NUM_CLASSES:
            confusion_mat[true_label][pred_label] += 1

    print(f"Total: {total}, Unknown Detected: {unknown_count}, Known: {total - unknown_count}")

    df_cm = pd.DataFrame(confusion_mat,
                         index=CLASS_NAMES,
                         columns=CLASS_NAMES + ['unknown'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix (with Unknown Class)")
    plt.tight_layout()
    plt.savefig(img_output_path)
    plt.close()

def visualize_features_pca(class_features):
    os.makedirs('t2', exist_ok=True)
    all_features = []
    labels = []
    for class_idx, feats in class_features.items():
        all_features.extend(feats)
        labels.extend([class_idx]*len(feats))

    all_features = np.array(all_features)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    for class_idx in range(len(class_features)):
        idxs = [i for i, l in enumerate(labels) if l == class_idx]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=CLASS_NAMES[class_idx], alpha=0.7)

    plt.legend()
    plt.title('Feature PCA Visualization')
    plt.savefig('t2/feature_pca.png')
    plt.close()

if __name__ == '__main__':
    alpha_value = 3  # ← 여기서 alpha 조정
    weibull_models, mean_vecs, class_features = fit_weibull(model, 'dataset_cls/train')
    visualize_features_pca(class_features)
    run_openmax_inference(TEST_IMAGE_DIR, TEST_LABEL_DIR, model, weibull_models, mean_vecs,
                          img_output_path='t2/confusion_matrix.png',
                          alpha=alpha_value)
    