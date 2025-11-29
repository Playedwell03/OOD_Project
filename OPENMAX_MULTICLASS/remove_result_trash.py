import os
import shutil

def remove_overlapping_files(folder1, folder2, output_dir):
    def get_basename_set(directory):
        return set(os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png', '.txt')))

    # 폴더 경로 설정
    folder1_images = os.path.join(folder1, 'images')
    folder1_labels = os.path.join(folder1, 'labels')
    folder2_files = get_basename_set(folder2)

    # 출력 경로 생성
    out_images = os.path.join(output_dir, 'images')
    out_labels = os.path.join(output_dir, 'labels')
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    count = 0

    # 이미지 복사
    for fname in os.listdir(folder1_images):
        basename, ext = os.path.splitext(fname)
        if basename not in folder2_files and ext.lower() in ['.jpg', '.jpeg', '.png']:
            shutil.copy(os.path.join(folder1_images, fname), os.path.join(out_images, fname))
            label_file = basename + '.txt'
            label_path = os.path.join(folder1_labels, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(out_labels, label_file))
            count += 1

    print(f"✅ 총 {count}개의 이미지/라벨 쌍이 output에 복사되었습니다 (겹치는 파일 제외).")

# 사용 예시
remove_overlapping_files(
    folder1='final_data_trash_removed',         # 원본 데이터 경로 (images/, labels/)
    folder2='clean_splitted/test/images',  # 비교 대상 폴더 (확장자 제외 이름 비교)
    output_dir='final_data_trash_removed_v2'       # 출력 경로
)