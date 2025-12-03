import os
import shutil
from collections import defaultdict
import random
import math

# --- (변경 없음) count_class_instances 함수 ---
def count_class_instances(label_dir):
    """지정된 디렉토리의 모든 라벨 파일에서 클래스별 인스턴스(객체) 수를 계산합니다."""
    class_counts = defaultdict(int)
    if not os.path.isdir(label_dir):
        return class_counts
        
    for filename in os.listdir(label_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(label_dir, filename), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            cls = int(parts[0])
                            class_counts[cls] += 1
                        except (ValueError, IndexError):
                            continue
    return class_counts

# --- (변경 없음) get_class0_files 함수 ---
def get_class0_files(ood_labels_dir):
    """OOD 데이터셋에서 클래스 0 객체만 포함하는 파일 목록을 반환합니다."""
    class0_files = []
    for filename in os.listdir(ood_labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(ood_labels_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                is_all_class0 = all(line.strip().startswith('0 ') for line in lines if line.strip())
                if lines and is_all_class0:
                    class0_files.append(filename)
            except Exception as e:
                print(f"파일 처리 중 오류 발생 {filename}: {e}")
    return class0_files

# --- (변경 없음) create_and_add_ood_from_class0 함수 ---
def create_and_add_ood_from_class0(base_dataset_dir, ood_dataset_dir, output_base_dir, ood_ratio=0.5, seed=42):
    """
    base_dataset의 각 클래스 비율에 맞춰 ood_dataset의 0번 클래스를 변형하여 OOD 데이터로 추가합니다.
    'valid' 및 'test' 스플릿은 OOD 추가 없이 그대로 복사합니다.
    """
    random.seed(seed)
    
    # --- 1. OOD 소스(클래스 0) 파일 목록 가져오기 ---
    split_train = 'train' 
    print(f"🔄 OOD 데이터 생성을 시작합니다... (대상: {base_dataset_dir})")

    ood_labels_dir = os.path.join(ood_dataset_dir, 'labels')
    ood_images_dir = os.path.join(ood_dataset_dir, 'images')
    
    print(f"🔍 '{ood_labels_dir}'에서 클래스 0 OOD 파일을 찾는 중...")
    class0_ood_files = get_class0_files(ood_labels_dir)
    print(f"  ➡️ 클래스 0 OOD 파일 {len(class0_ood_files)}개 발견")
    if not class0_ood_files:
        print("⚠️ OOD로 사용할 클래스 0 파일이 없습니다. 'train' 작업을 중단합니다.")
    
    # --- 2. 출력 디렉토리 설정 및 기존 'train' 데이터 복사 ---
    out_labels_dir = os.path.join(output_base_dir, split_train, 'labels')
    out_images_dir = os.path.join(output_base_dir, split_train, 'images')
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    src_labels_dir = os.path.join(base_dataset_dir, split_train, 'labels')
    src_images_dir = os.path.join(base_dataset_dir, split_train, 'images')

    print(f"💾 '{src_labels_dir}'의 원본 'train' 데이터를 '{out_labels_dir}'로 복사 중...")
    if os.path.exists(src_labels_dir):
        for fname in os.listdir(src_labels_dir):
            shutil.copy(os.path.join(src_labels_dir, fname), out_labels_dir)
            
    if os.path.exists(src_images_dir):
        for fname in os.listdir(src_images_dir):
             shutil.copy(os.path.join(src_images_dir, fname), out_images_dir)

    # --- 3. 원본 'train' 데이터의 클래스별 인스턴스 수 계산 ---
    print("📊 원본 'train' 데이터셋의 클래스별 인스턴스 수 계산 중...")
    original_class_counts = count_class_instances(src_labels_dir)
    print("  ", sorted(original_class_counts.items()))

    # --- 4. 클래스별로 OOD 데이터 생성 및 'train'에 추가 ---
    if class0_ood_files:
        used_ood_files = set() 
        
        for cls, count in sorted(original_class_counts.items()):
            if count == 0:
                continue
                
            num_ood_needed = math.ceil(count * ood_ratio / (1 - ood_ratio))
            available_files = [f for f in class0_ood_files if f not in used_ood_files]

            if not available_files:
                print(f"  ⚠️ 클래스 {cls}: 추가할 수 있는 OOD 파일이 더 이상 없습니다. 중단합니다.")
                break
            
            if len(available_files) < num_ood_needed:
                print(f"  ⚠️ 클래스 {cls}: 필요한 {num_ood_needed}개보다 OOD 파일이 부족하여 {len(available_files)}개만 사용합니다.")
                selected_files = available_files
            else:
                selected_files = random.sample(available_files, num_ood_needed)

            print(f"  ✅ 클래스 {cls}: {count}개 인스턴스 기준 -> {len(selected_files)}개의 클래스 0 파일을 클래스 {cls}(으)로 변환하여 추가")

            for fname in selected_files:
                used_ood_files.add(fname) 

                img_name = os.path.splitext(fname)[0] + '.jpg'
                img_src = os.path.join(ood_images_dir, img_name)
                img_dst = os.path.join(out_images_dir, img_name)
                if os.path.exists(img_src):
                    shutil.copy(img_src, img_dst)
                else:
                     print(f"    ⚠️ 이미지 없음: {img_src}")

                label_src = os.path.join(ood_labels_dir, fname)
                label_dst = os.path.join(out_labels_dir, fname)
                
                with open(label_src, 'r') as infile, open(label_dst, 'w') as outfile:
                    for line in infile:
                        parts = line.strip().split()
                        if parts:
                            parts[0] = str(cls) 
                            new_line = ' '.join(parts) + '\n'
                            outfile.write(new_line)
                            
        print("\n✅ OOD 데이터 'train' 스플릿에 추가 작업 완료!")
    
    final_class_counts = count_class_instances(out_labels_dir)
    print("📊 최종 'train' 스플릿의 클래스별 인스턴스 수:")
    print("  ", sorted(final_class_counts.items()))

    # --- 5. [추가됨] 'valid' 및 'test' 폴더 복사 (OOD 추가 없음) ---
    print("\n🔄 'valid' 및 'test' 폴더 복사를 시작합니다...")
    
    for split_name in ['valid', 'test']:
        src_dir = os.path.join(base_dataset_dir, split_name)
        dst_dir = os.path.join(output_base_dir, split_name)
        
        if os.path.isdir(src_dir):
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            print(f"  💾 '{src_dir}'를 '{dst_dir}'로 복사 중...")
            shutil.copytree(src_dir, dst_dir)
        else:
            print(f"  ℹ️ '{src_dir}' 폴더가 없어 복사를 건너뜁니다.")
            
    print(f"✅ 'valid'/'test' 폴더 복사 완료. ({output_base_dir})")

# -----------------------------------------------------------------
# --- [수정됨] K-Fold 자동화 래퍼 (Wrapper) ---
# -----------------------------------------------------------------
K = 10
SEED = 42

# --- ⚠️ 여기만 수정하여 사용하세요 ---
RATIO_TO_ADD = 0.9  # 10% = 0.1, 30% = 0.3, 50% = 0.5
FINAL_OUTPUT_DIR = 'A_k10_runs_90per' # 저장할 최종 부모 폴더 이름
# ---

# 고정 경로
BASE_RUNS_DIR = 'A_k10_runs'             # K-Fold 10개 원본
OOD_SOURCE_DIR = 'fish_class_0'          # OOD 소스

print("="*60)
print(f"K-Fold OOD 추가 자동화를 시작합니다 (K={K}, Ratio={RATIO_TO_ADD})")
print(f"결과 저장 위치: {FINAL_OUTPUT_DIR}")
print("="*60)

for i in range(1, K + 1):
    run_name = f'run_{i}_test'
    
    base_dir = os.path.join(BASE_RUNS_DIR, run_name)
    output_dir = os.path.join(FINAL_OUTPUT_DIR, run_name)
    
    print(f"\n--- [ {i}/{K} ] 작업 시작: {run_name} ---")
    
    create_and_add_ood_from_class0(
        base_dataset_dir=base_dir,
        ood_dataset_dir=OOD_SOURCE_DIR,
        output_base_dir=output_dir,
        ood_ratio=RATIO_TO_ADD,
        seed=SEED
    )
    
print("\n" + "="*60)
print("🎉 모든 K-Fold Run에 대한 OOD 추가 작업이 완료되었습니다.")
print("="*60)