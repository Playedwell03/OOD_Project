import os
import shutil
from collections import defaultdict
import random
import math

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

def get_class0_files(ood_labels_dir):
    """OOD 데이터셋에서 클래스 0 객체만 포함하는 파일 목록을 반환합니다."""
    class0_files = []
    for filename in os.listdir(ood_labels_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(ood_labels_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                # 모든 줄의 클래스가 0인지 확인
                is_all_class0 = all(line.strip().startswith('0 ') for line in lines if line.strip())
                if lines and is_all_class0:
                    class0_files.append(filename)
            except Exception as e:
                print(f"파일 처리 중 오류 발생 {filename}: {e}")
    return class0_files

def create_and_add_ood_from_class0(base_dataset_dir, ood_dataset_dir, output_base_dir, ood_ratio=0.5, seed=42):
    """
    base_dataset의 각 클래스 비율에 맞춰 ood_dataset의 0번 클래스를 변형하여 OOD 데이터로 추가합니다.
    """
    random.seed(seed)
    split = 'train'  # 'train' 스플릿에 대해서만 작업을 수행합니다.

    print("🔄 OOD 데이터 생성을 시작합니다...")

    # 1. OOD 소스(클래스 0) 파일 목록 가져오기
    ood_labels_dir = os.path.join(ood_dataset_dir, 'labels')
    ood_images_dir = os.path.join(ood_dataset_dir, 'images')
    
    print(f"🔍 '{ood_labels_dir}'에서 클래스 0 OOD 파일을 찾는 중...")
    class0_ood_files = get_class0_files(ood_labels_dir)
    print(f"  ➡️ 클래스 0 OOD 파일 {len(class0_ood_files)}개 발견")
    if not class0_ood_files:
        print("⚠️ OOD로 사용할 클래스 0 파일이 없습니다. 작업을 중단합니다.")
        return

    # 2. 출력 디렉토리 설정 및 기존 'train' 데이터 복사
    out_labels_dir = os.path.join(output_base_dir, split, 'labels')
    out_images_dir = os.path.join(output_base_dir, split, 'images')
    os.makedirs(out_labels_dir, exist_ok=True)
    os.makedirs(out_images_dir, exist_ok=True)

    src_labels_dir = os.path.join(base_dataset_dir, split, 'labels')
    src_images_dir = os.path.join(base_dataset_dir, split, 'images')

    print(f"💾 '{src_labels_dir}'의 원본 데이터를 '{out_labels_dir}'로 복사 중...")
    if os.path.exists(src_labels_dir):
        for fname in os.listdir(src_labels_dir):
            shutil.copy(os.path.join(src_labels_dir, fname), out_labels_dir)
            
    if os.path.exists(src_images_dir):
        for fname in os.listdir(src_images_dir):
             shutil.copy(os.path.join(src_images_dir, fname), out_images_dir)

    # 3. 원본 데이터의 클래스별 인스턴스 수 계산
    print("📊 원본 데이터셋의 클래스별 인스턴스 수 계산 중...")
    original_class_counts = count_class_instances(src_labels_dir)
    print("  ", sorted(original_class_counts.items()))

    # 4. 클래스별로 OOD 데이터 생성 및 추가
    used_ood_files = set() # 한 번 사용한 OOD 파일은 다시 사용하지 않도록 관리
    
    # 클래스 번호 순으로 정렬하여 처리
    for cls, count in sorted(original_class_counts.items()):
        if count == 0:
            continue
            
        # 추가해야 할 OOD 인스턴스 수 계산
        # 최종 데이터셋에서 OOD 비율이 ood_ratio가 되도록 계산합니다.
        # N_ood = (N_orig * ratio) / (1 - ratio)
        num_ood_needed = math.ceil(count * ood_ratio / (1 - ood_ratio))
        
        # 사용 가능한 OOD 파일 목록
        available_files = [f for f in class0_ood_files if f not in used_ood_files]

        if not available_files:
            print(f"  ⚠️ 클래스 {cls}: 추가할 수 있는 OOD 파일이 더 이상 없습니다. 중단합니다.")
            break
        
        # 필요한 만큼 파일 샘플링
        if len(available_files) < num_ood_needed:
            print(f"  ⚠️ 클래스 {cls}: 필요한 {num_ood_needed}개보다 OOD 파일이 부족하여 {len(available_files)}개만 사용합니다.")
            selected_files = available_files
        else:
            selected_files = random.sample(available_files, num_ood_needed)

        print(f"  ✅ 클래스 {cls}: {count}개 인스턴스 기준 -> {len(selected_files)}개의 클래스 0 파일을 클래스 {cls}(으)로 변환하여 추가")

        for fname in selected_files:
            used_ood_files.add(fname) # 사용 처리

            # 이미지 복사
            img_name = os.path.splitext(fname)[0] + '.jpg'
            img_src = os.path.join(ood_images_dir, img_name)
            img_dst = os.path.join(out_images_dir, img_name)
            if os.path.exists(img_src):
                shutil.copy(img_src, img_dst)
            else:
                 print(f"    ⚠️ 이미지 없음: {img_src}")

            # 라벨 변환 및 복사
            label_src = os.path.join(ood_labels_dir, fname)
            label_dst = os.path.join(out_labels_dir, fname)
            
            with open(label_src, 'r') as infile, open(label_dst, 'w') as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if parts:
                        # 클래스 ID를 현재 목표 클래스(cls)로 변경
                        parts[0] = str(cls)
                        new_line = ' '.join(parts) + '\n'
                        outfile.write(new_line)
                        
    print("\n✅ OOD 데이터 추가 작업 완료!")
    final_class_counts = count_class_instances(out_labels_dir)
    print("📊 최종 데이터셋의 클래스별 인스턴스 수:")
    print("  ", sorted(final_class_counts.items()))

# --- 사용 예시 ---
# 아래 경로들을 실제 환경에 맞게 수정하여 사용하세요.
create_and_add_ood_from_class0(
    base_dataset_dir='A_k10_runs/run_1_test',              # 원본 데이터셋 폴더
    ood_dataset_dir='fish_class_0',     # 클래스 0 데이터를 가져올 OOD 데이터셋 폴더
    output_base_dir='A_k10_runs_30per/run_1_test',  # 결과물을 저장할 폴더
    ood_ratio=0.3                          # 최종 데이터셋에서 OOD가 차지할 비율 (0.5 = 50%)
)