import os
import shutil
import sys

# --- 1. 사용자 설정 ---

# 10개 폴더(fold_1 ~ fold_10)가 들어있는 기본 폴더
# (이전 스크립트의 output_dir)
INPUT_BASE_DIR = 'A_k10_folds'

# 10개의 완전한 데이터셋(run_1_test ~ run_10_test)을 생성할 위치
OUTPUT_RUNS_DIR = 'A_k10_runs_10per'

K = 10  # 10-fold

# --- 2. 헬퍼 함수 (폴더 안의 모든 파일 복사) ---
def copy_folder_contents(src_folder, dst_folder):
    """
    src_folder(예: fold_1) 안의 'images'와 'labels' 폴더 내용을
    dst_folder(예: run_10_test/train) 안의 'images'와 'labels'로 복사
    """
    # images 폴더 복사
    src_images = os.path.join(src_folder, 'images')
    dst_images = os.path.join(dst_folder, 'images')
    if os.path.exists(src_images):
        os.makedirs(dst_images, exist_ok=True)
        for item in os.listdir(src_images):
            shutil.copy(os.path.join(src_images, item), dst_images)
            
    # labels 폴더 복사
    src_labels = os.path.join(src_folder, 'labels')
    dst_labels = os.path.join(dst_folder, 'labels')
    if os.path.exists(src_labels):
        os.makedirs(dst_labels, exist_ok=True)
        for item in os.listdir(src_labels):
            shutil.copy(os.path.join(src_labels, item), dst_labels)

# --- 3. 설정 및 검증 ---
os.makedirs(OUTPUT_RUNS_DIR, exist_ok=True)
all_fold_nums = list(range(1, K + 1))
print(f"--- {K}-Fold (8:1:1) Physical Dataset Copier ---")
print(f"Input Folds:   {INPUT_BASE_DIR}")
print(f"Output Datasets: {OUTPUT_RUNS_DIR}\n")

if not os.path.exists(os.path.join(INPUT_BASE_DIR, 'fold_1')):
    print(f"❌ ERROR: '{INPUT_BASE_DIR}/fold_1' 폴더를 찾을 수 없습니다.")
    print("이전 스크립트를 먼저 실행해서 fold_1 ~ fold_10 폴더를 생성해야 합니다.")
    sys.exit()

# --- 4. 10번의 실험(Run) 루프 ---
for test_fold_num in all_fold_nums:
    
    # 4.1. Train / Valid / Test 폴더 인덱스 결정
    if test_fold_num == 1:
        valid_fold_num = K
    else:
        valid_fold_num = test_fold_num - 1
    
    train_fold_nums = []
    for num in all_fold_nums:
        if num != test_fold_num and num != valid_fold_num:
            train_fold_nums.append(num)
            
    # 4.2. 님이 요청한 "fold_n_test" 네이밍 반영
    run_name = f"run_{test_fold_num}_test" # (run_1_test, run_2_test, ...)
    run_output_dir = os.path.join(OUTPUT_RUNS_DIR, run_name)
    
    print(f"--- Creating Dataset: {run_name} (Test on fold {test_fold_num:02d}) ---")
    
    # 4.3. 이 Run에 필요한 3개의 대상 폴더(train/valid/test) 생성
    train_dir = os.path.join(run_output_dir, 'train')
    valid_dir = os.path.join(run_output_dir, 'valid')
    test_dir  = os.path.join(run_output_dir, 'test')
    
    # (주의) 혹시 이미 폴더가 있다면 삭제하고 새로 만듭니다.
    if os.path.exists(run_output_dir):
        shutil.rmtree(run_output_dir)
        
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 4.4. 파일 복사 실행

    # Train (8개 폴더 합치기)
    print(f"  Copying Train folds: {train_fold_nums} -> {run_name}/train/")
    for fold_num in train_fold_nums:
        src_fold = os.path.join(INPUT_BASE_DIR, f'fold_{fold_num}')
        copy_folder_contents(src_fold, train_dir)

    # Valid (1개 폴더 복사)
    print(f"  Copying Valid fold:  [{valid_fold_num}] -> {run_name}/valid/")
    src_fold = os.path.join(INPUT_BASE_DIR, f'fold_{valid_fold_num}')
    copy_folder_contents(src_fold, valid_dir)
    
    # Test (1S개 폴더 복사)
    print(f"  Copying Test fold:   [{test_fold_num}] -> {run_name}/test/")
    src_fold = os.path.join(INPUT_BASE_DIR, f'fold_{test_fold_num}')
    copy_folder_contents(src_fold, test_dir)
    
    print(f"  -> Dataset '{run_name}' 생성 완료.\n")

print("\n✅ 모든 10개 Run의 데이터셋 복사/생성을 완료했습니다.")