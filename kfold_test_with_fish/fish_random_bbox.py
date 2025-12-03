import os
import random
import shutil
from tqdm import tqdm

def randomize_labels_in_directory(
    source_label_dir, 
    target_label_dir, 
    seed=42
):
    """
    source_label_dir의 라벨 파일을 읽어서,
    클래스 ID는 유지하되 bbox 좌표(x, y, w, h)를 랜덤하게 변경하여
    target_label_dir에 저장합니다.
    """
    random.seed(seed)
    os.makedirs(target_label_dir, exist_ok=True)
    
    files = [f for f in os.listdir(source_label_dir) if f.endswith('.txt')]
    print(f"🔄 랜덤 BBox 생성 시작: {len(files)}개 파일 처리 중...")

    for fname in tqdm(files):
        src_path = os.path.join(source_label_dir, fname)
        dst_path = os.path.join(target_label_dir, fname)
        
        new_lines = []
        with open(src_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            class_id = parts[0] # 클래스 ID는 유지 (또는 0으로 고정)
            
            # --- [핵심] 좌표 랜덤 생성 (YOLO format: 0.0 ~ 1.0) ---
            # 너비와 높이를 먼저 랜덤하게 정함 (너무 작거나 크지 않게 0.05 ~ 0.5 사이)
            new_w = random.uniform(0.05, 0.5)
            new_h = random.uniform(0.05, 0.5)
            
            # 중심 좌표(x, y)는 박스가 이미지 밖으로 나가지 않게 범위 설정
            # min_x = w/2, max_x = 1 - w/2
            new_x = random.uniform(new_w / 2, 1.0 - new_w / 2)
            new_y = random.uniform(new_h / 2, 1.0 - new_h / 2)
            
            # 새로운 라인 생성
            new_line = f"{class_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n"
            new_lines.append(new_line)
            
        # 저장
        with open(dst_path, 'w') as f:
            f.writelines(new_lines)

    print(f"✅ 랜덤 BBox 라벨 생성 완료: {target_label_dir}")

# --- 사용 예시 ---

# 1. 원본 OOD 라벨 폴더 (물고기 정답 라벨)
ORIGINAL_OOD_LABELS = 'fish_class_0/labels'

# 2. 새로 만들 랜덤 OOD 라벨 폴더
RANDOM_OOD_LABELS = 'fish_random_bbox/labels'

# 3. 실행
randomize_labels_in_directory(ORIGINAL_OOD_LABELS, RANDOM_OOD_LABELS)

# 4. (중요) 이미지 폴더는 원본 그대로 복사해주거나 심볼릭 링크를 걸어야 합니다.
#    YOLO 학습을 위해 'fish_random_bbox/images' 폴더도 필요합니다.
#    여기서는 편의상 원본 이미지를 복사하는 코드를 추가합니다.
original_img_dir = 'fish_class_0/images'
target_img_dir = 'fish_random_bbox/images'

print("📂 이미지 파일 복사 중... (시간이 좀 걸릴 수 있습니다)")
if os.path.exists(target_img_dir):
    shutil.rmtree(target_img_dir)
shutil.copytree(original_img_dir, target_img_dir)
print("✅ 이미지 복사 완료.")