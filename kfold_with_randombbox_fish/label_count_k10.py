import os
import yaml
import matplotlib.pyplot as plt
from collections import Counter

# --- 1. 사용자 설정 ---
# 분석할 최상위 데이터셋 폴더
BASE_DATA_DIR = 'A_k10_runs' 

# 클래스 이름을 가져올 참조용 YAML 파일 경로 
# (모든 Fold가 같은 클래스를 공유하므로 하나만 지정해도 됩니다)
# 만약 각 폴더 안에 data.yaml이 있다면 그것을 읽도록 수정할 수도 있습니다.
REF_YAML_PATH = 'A_k10_runs/run_1_test/data.yaml' 

# K-Fold 횟수
K = 10 

# --- 2. 클래스 이름 로드 ---
if os.path.exists(REF_YAML_PATH):
    with open(REF_YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    class_names = data['names']
    print(f"✅ 클래스 이름을 '{REF_YAML_PATH}'에서 로드했습니다. (총 {len(class_names)}개)")
else:
    print(f"❌ YAML 파일을 찾을 수 없습니다: {REF_YAML_PATH}")
    # 임시 클래스 이름 (필요시 수정)
    class_names = [str(i) for i in range(16)] 

# --- 3. 분석 및 시각화 함수 ---
def process_and_plot(label_dir, output_dir, title_suffix):
    if not os.path.exists(label_dir):
        print(f"  ⚠️ 라벨 폴더 없음: {label_dir}")
        return

    # 라벨 개수 세기
    label_counts = Counter()
    file_count = 0
    
    for file_name in os.listdir(label_dir):
        if file_name.endswith('.txt'):
            file_count += 1
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            class_index = int(line.split()[0])
                            label_counts[class_index] += 1
                        except ValueError:
                            continue

    # 결과 정리
    counts = [label_counts[i] for i in range(len(class_names))]
    total_labels = sum(counts)

    # --- 시각화 ---
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, counts, color='skyblue')
    
    plt.xlabel('Class')
    plt.ylabel('Label Count')
    plt.title(f'Label Count per Class - {title_suffix}')
    plt.xticks(rotation=45, ha='right') # 클래스 이름이 길 경우 기울임

    # 막대 위에 개수 표시
    for bar, count in zip(bars, counts):
        if count > 0: # 0개인 경우 표시 안 함 (깔끔하게)
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(count), ha='center', va='bottom', fontsize=9)

    # 전체 라벨 수 및 파일 수 표시
    info_text = f'Total Files: {file_count}\nTotal Labels: {total_labels}'
    plt.text(0.98, 0.95, info_text, transform=plt.gca().transAxes,
             ha='right', va='top', fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    plt.tight_layout()

    # 저장
    output_path = os.path.join(output_dir, 'label_distribution.png')
    plt.savefig(output_path)
    plt.close() # 메모리 해제
    
    print(f"  ✅ 저장 완료: {output_path} (Labels: {total_labels})")


# --- 4. 메인 루프 (K=1 ~ 10, Train/Valid/Test) ---
print(f"\n🚀 전체 데이터셋 라벨 분포 분석 시작 ({BASE_DATA_DIR})...\n")

for i in range(1, K + 1):
    run_name = f'run_{i}_test'
    print(f"--- Processing {run_name} ---")
    
    # 각 Split(train, valid, test)에 대해 반복
    for split in ['train']:
        target_dir = os.path.join(BASE_DATA_DIR, run_name, split)
        label_dir = os.path.join(target_dir, 'labels')
        
        # 시각화 함수 호출
        # output_dir은 이미지가 저장될 경로 (label_dir의 상위인 train/valid/test 폴더)
        process_and_plot(label_dir, target_dir, f"{run_name} [{split}]")

print("\n🎉 모든 그래프 생성이 완료되었습니다.")