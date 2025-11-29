import os
import cv2
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# --- 1. 시각화 헬퍼 함수 ---
def save_visualization(img_path, label_lines, save_dir_map, class_names, figsize=(10, 10)):
    """
    이미지에 BBox를 그리고, 해당 객체가 포함된 클래스 폴더들에 저장합니다.
    save_dir_map: {class_id: [save_path_1, save_path_2, ...]}
    """
    try:
        img = cv2.imread(img_path)
        if img is None: return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
    except Exception:
        return

    # 플롯 생성
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)
    ax.axis('off')

    # BBox 그리기
    for line in label_lines:
        parts = line.strip().split()
        cls_idx = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        label_text = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)

        if len(coords) == 4: # YOLO BBox
            cx, cy, bw, bh = coords
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            rect = plt.Rectangle((x1, y1), bw * w, bh * h,
                                 edgecolor='lime', facecolor='none', linewidth=2)
            ax.add_patch(rect)
            ax.text(x1, y1 - 5, label_text, color='white',
                    bbox=dict(facecolor='green', alpha=0.5, edgecolor='none'), fontsize=8)
        
        elif len(coords) >= 6: # Polygon
            pts = np.array(coords).reshape(-1, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            poly = plt.Polygon(pts, edgecolor='orange', facecolor='none', linewidth=2)
            ax.add_patch(poly)

    # 저장 (해당 이미지에 등장한 모든 클래스의 폴더에 각각 저장)
    # save_dir_map에는 이 이미지에 있는 클래스들에 대한 저장 경로가 들어있음
    saved_paths = set()
    for cls_id, paths in save_dir_map.items():
        for path in paths:
            if path in saved_paths: continue # 중복 저장 방지
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            saved_paths.add(path)
    
    plt.close(fig)

# --- 2. 메인 분석 함수 ---
def analyze_and_visualize_dataset(
    mixed_data_dir, 
    normal_source_dir, 
    ood_source_dir, 
    output_vis_dir, 
    class_names
):
    print("📂 원본 데이터 파일 목록 인덱싱 중...")
    
    def get_filenames(folder):
        target = os.path.join(folder, 'labels') if os.path.exists(os.path.join(folder, 'labels')) else folder
        return set([f for f in os.listdir(target) if f.endswith('.txt')])

    normal_files = get_filenames(normal_source_dir)
    ood_files = get_filenames(ood_source_dir)
    
    mixed_labels_dir = os.path.join(mixed_data_dir, 'labels')
    mixed_images_dir = os.path.join(mixed_data_dir, 'images')
    
    if not os.path.exists(mixed_labels_dir):
        print(f"❌ 오류: 라벨 폴더 없음: {mixed_labels_dir}")
        return

    # 통계 변수 (객체 수 기준)
    # class_stats[class_id]['normal'] = 해당 클래스의 정상 객체 수
    class_stats = defaultdict(lambda: {'normal': 0, 'ood': 0})
    
    files_processed = 0
    print(f"\n🔍 분석 및 시각화 저장 시작... (저장 위치: {output_vis_dir})")

    # 파일 순회
    for fname in os.listdir(mixed_labels_dir):
        if not fname.endswith('.txt'): continue
        
        files_processed += 1
        label_path = os.path.join(mixed_labels_dir, fname)
        
        # 출처 판별 (파일명 기준)
        source_type = "Unknown"
        if fname in normal_files: source_type = "Normal"
        elif fname in ood_files: source_type = "OOD"
        
        if source_type == "Unknown": continue

        # 이미지 파일 찾기
        img_name_base = os.path.splitext(fname)[0]
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            temp_path = os.path.join(mixed_images_dir, img_name_base + ext)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
        
        if img_path is None: continue

        # 라벨 파일 읽기
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        # --- [핵심] 객체 수 카운트 및 저장 경로 설정 ---
        save_dir_map = defaultdict(list) # {cls_id: [저장경로1, 저장경로2]}
        
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            
            # 1. 통계 카운트 (줄 단위 = 객체 단위)
            if source_type == "Normal":
                class_stats[cls_id]['normal'] += 1
            else:
                class_stats[cls_id]['ood'] += 1
            
            # 2. 저장 경로 예약 (이미지 하나에 같은 클래스가 여러 개면 한 번만 저장하도록)
            if cls_id < len(class_names):
                cls_name = class_names[cls_id]
                save_path = os.path.join(output_vis_dir, cls_name, source_type, os.path.basename(img_path))
                # 리스트에 없으면 추가
                if save_path not in save_dir_map[cls_id]:
                    save_dir_map[cls_id].append(save_path)

        # 시각화 및 저장 실행
        if save_dir_map:
            save_visualization(img_path, lines, save_dir_map, class_names)

        if files_processed % 100 == 0:
            print(f"   ... {files_processed}개 파일 처리 중")

    # --- 3. 결과 출력 (객체 수 기준) ---
    print(f"\n📊 분석 완료 (총 처리 파일 수: {files_processed})")
    print("※ 아래 표는 파일 개수가 아니라 '객체(Instance/Label) 개수'입니다.")
    print("-" * 80)
    print(f"{'Class Name':<20} | {'Normal Inst.':<15} | {'OOD Inst.':<15} | {'OOD Ratio':<10}")
    print("-" * 80)
    
    sorted_ids = sorted(class_stats.keys())
    total_normal = 0
    total_ood = 0
    
    for cls_id in sorted_ids:
        if cls_id < len(class_names):
            name = class_names[cls_id]
            n = class_stats[cls_id]['normal']
            o = class_stats[cls_id]['ood']
            total = n + o
            ratio = (o/total*100) if total > 0 else 0
            
            total_normal += n
            total_ood += o
            print(f"{name:<20} | {n:<15} | {o:<15} | {ratio:.1f}%")

    print("-" * 80)
    tot_all = total_normal + total_ood
    tot_ratio = (total_ood/tot_all*100) if tot_all > 0 else 0
    print(f"{'TOTAL (Objects)':<20} | {total_normal:<15} | {total_ood:<15} | {tot_ratio:.1f}%")
    print("-" * 80)

# --- 실행 설정 ---

CLASS_NAMES = [
    'crackdown', 'crossline', 'danger', 'kidprotectzone', 'leftcurve', 
    'noparking', 'nouturn', 'ntrack', 'oneway', 'slipper', 'slow', 
    'speedbump', 'speedlimit', 'stop', 'turncross', 'uturn'
]

# 경로 설정 (본인 경로에 맞게 수정)
MIXED_PATH = r'added_fish_A_ver2/50per/train'      # 분석할 50% 데이터셋
NORMAL_SRC = r'A/train'             # 원본 정상 데이터셋
OOD_SRC = r'fish_class_0'                               # 원본 OOD 데이터셋
OUTPUT_VIS = r'visualization_output_50per_bbox'         # 결과 저장 폴더

# 실행
analyze_and_visualize_dataset(
    mixed_data_dir=MIXED_PATH,
    normal_source_dir=NORMAL_SRC,
    ood_source_dir=OOD_SRC,
    output_vis_dir=OUTPUT_VIS,
    class_names=CLASS_NAMES
)