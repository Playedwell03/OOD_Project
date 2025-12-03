import os
import glob
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def visualize_yolo_dataset(
    images_dir,
    labels_dir,
    output_dir,
    class_names=None,
    num_samples=None  # None이면 전체 변환, 숫자를 넣으면 그만큼만 랜덤 샘플링
):
    """
    YOLO 포맷의 데이터셋(이미지+라벨)을 읽어 BBox를 시각화하고 저장합니다.
    """
    
    # 1. 이미지 파일 목록 가져오기
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # 샘플링 (전체가 너무 많을 경우를 대비)
    if num_samples and num_samples < len(image_paths):
        image_paths = random.sample(image_paths, num_samples)
        print(f"🎲 전체 {len(image_paths)}장 중 {num_samples}장을 랜덤 샘플링했습니다.")
    else:
        print(f"📂 전체 {len(image_paths)}장 이미지를 처리합니다.")

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 2. 색상 팔레트 생성 (클래스별 고유 색상)
    # matplotlib의 tab20 컬러맵 사용
    cmap = plt.get_cmap('tab20')
    colors = [tuple(np.array(cmap(i)[:3]) * 255) for i in range(20)]

    count = 0
    for img_path in image_paths:
        # 2.1. 이미지 로드
        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            draw = ImageDraw.Draw(img)
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {os.path.basename(img_path)} ({e})")
            continue

        # 2.2. 대응하는 라벨 파일 찾기
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, basename + ".txt")

        if not os.path.exists(label_path):
            # 라벨이 없으면 원본 이미지만 저장 (또는 건너뛰기)
            # print(f"⚠️ 라벨 파일 없음: {basename}.txt")
            continue

        # 2.3. 라벨 읽기 및 그리기
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue

            cls_idx = int(parts[0])
            # YOLO Format: center_x, center_y, width, height (Normalized 0~1)
            cx, cy, bw, bh = map(float, parts[1:5])

            # 좌표 변환 (Pixel)
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            # 클래스 정보
            label_text = str(cls_idx)
            if class_names and cls_idx < len(class_names):
                label_text = class_names[cls_idx]

            # 색상 선택
            color = colors[cls_idx % len(colors)]
            color_int = (int(color[0]), int(color[1]), int(color[2]))

            # 박스 그리기 (두께 3)
            draw.rectangle([x1, y1, x2, y2], outline=color_int, width=3)
            
            # 텍스트 배경 및 텍스트 그리기
            # (폰트가 없으면 기본 폰트 사용)
            try:
                # 폰트 크기를 이미지 크기에 비례하게 설정하거나 고정값 사용
                font_size = max(12, int(h / 40)) 
                # 윈도우의 경우 arial.ttf, 리눅스는 DejaVuSans.ttf 등을 시도해볼 수 있음
                # 여기선 기본 load_default() 사용 (영문만 가능할 수 있음)
                font = ImageFont.load_default() 
            except:
                font = None
            
            # 텍스트 박스 크기 계산 (대략적)
            text_w = len(label_text) * 6
            text_h = 12
            
            draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 4, y1], fill=color_int)
            draw.text((x1 + 2, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)

        # 2.4. 저장
        save_path = os.path.join(output_dir, os.path.basename(img_path))
        img.save(save_path)
        count += 1
        
        if count % 100 == 0:
            print(f"   ... {count}장 처리 완료")

    print(f"✅ 시각화 완료! 결과가 '{output_dir}'에 저장되었습니다.")


# --- ⚠️ 사용자 설정 ---

# 1. 클래스 이름 (라벨 ID 순서대로)
MY_CLASSES = [
    'crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 
    'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 
    'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn'
]

# 2. 경로 설정 (확인하고 싶은 데이터셋 경로를 입력하세요)
TARGET_IMAGES_DIR = 'fish_random_bbox/images'  # 예: 랜덤 BBox 이미지 폴더
TARGET_LABELS_DIR = 'fish_random_bbox/labels'  # 예: 랜덤 BBox 라벨 폴더

# 3. 결과 저장 경로
SAVE_VIS_DIR = 'vis_output_random_bbox'

# --- 실행 ---
visualize_yolo_dataset(
    images_dir=TARGET_IMAGES_DIR,
    labels_dir=TARGET_LABELS_DIR,
    output_dir=SAVE_VIS_DIR,
    class_names=MY_CLASSES,
    num_samples=50  # 50장만 랜덤으로 뽑아서 확인 (전체를 보려면 None)
)