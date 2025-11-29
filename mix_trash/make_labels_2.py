import os
import shutil

def filter_yolo_data_and_images(
    image_input_dir, 
    label_input_dir, 
    label_output_dir, 
    image_output_dir
):
    """
    YOLO 라벨을 필터링하고, 그에 해당하는 이미지 파일을 새 폴더로 복사합니다.
    """
    # -------------------------------------------------------------------
    # 작업 1: 라벨 파일 필터링 (클래스 1, 5만 남기기)
    # -------------------------------------------------------------------
    os.makedirs(label_output_dir, exist_ok=True)
    print(f"[작업 1] 라벨 필터링 시작...")
    print(f"✅ 필터링된 라벨은 '{label_output_dir}' 폴더에 저장됩니다.")

    processed_labels = 0
    saved_labels = 0

    for filename in os.listdir(label_input_dir):
        if filename.endswith(".txt"):
            processed_labels += 1
            input_filepath = os.path.join(label_input_dir, filename)
            
            kept_lines = []
            try:
                with open(input_filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        stripped_line = line.strip()
                        if stripped_line:
                            parts = stripped_line.split()
                            if parts and int(parts[0]) in [1, 5]:
                                kept_lines.append(stripped_line)
            except Exception as e:
                print(f"⚠️ '{filename}' 라벨 파일을 읽는 중 오류 발생: {e}")
                continue

            if kept_lines:
                saved_labels += 1
                output_filepath = os.path.join(label_output_dir, filename)
                try:
                    with open(output_filepath, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(kept_lines) + '\n')
                except Exception as e:
                    print(f"⚠️ '{filename}' 라벨 파일을 쓰는 중 오류 발생: {e}")
    
    print(f"➡️ 라벨 필터링 완료: {processed_labels}개 파일 중 {saved_labels}개를 저장했습니다.")
    print("-" * 40)

    # -------------------------------------------------------------------
    # 작업 2: 필터링된 라벨과 이름이 같은 이미지 파일 복사
    # -------------------------------------------------------------------
    print(f"[작업 2] 이미지 복사 시작...")
    os.makedirs(image_output_dir, exist_ok=True)
    print(f"✅ 필터링된 이미지는 '{image_output_dir}' 폴더에 저장됩니다.")

    # 저장된 라벨 파일들의 이름(확장자 제외)을 집합(set)으로 만듭니다. (빠른 조회를 위함)
    final_label_basenames = {os.path.splitext(f)[0] for f in os.listdir(label_output_dir)}

    if not final_label_basenames:
        print("⚠️ 필터링된 라벨이 없어 이미지 복사를 진행하지 않습니다.")
        return

    copied_images = 0
    # 원본 이미지 폴더를 순회하며 짝이 맞는 파일을 찾습니다.
    for image_filename in os.listdir(image_input_dir):
        image_basename = os.path.splitext(image_filename)[0]
        
        # 이미지 파일명이 필터링된 라벨 목록에 있는지 확인
        if image_basename in final_label_basenames:
            source_path = os.path.join(image_input_dir, image_filename)
            dest_path = os.path.join(image_output_dir, image_filename)
            
            try:
                # 이미지 파일을 새 폴더로 복사합니다.
                shutil.copy2(source_path, dest_path)
                copied_images += 1
            except Exception as e:
                print(f"⚠️ '{image_filename}' 이미지 파일을 복사하는 중 오류 발생: {e}")

    print(f"➡️ 이미지 복사 완료: {copied_images}개의 이미지 파일을 복사했습니다.")
    print("-" * 40)
    print("🎉 모든 작업이 성공적으로 완료되었습니다!")


# ===================================================================
# 사용법: 아래 4개 변수의 경로를 실제 환경에 맞게 수정해주세요.
# ===================================================================

# 1. 원본 이미지 파일들이 들어있는 폴더
# 예: image_input_folder = 'D:/datasets/original/images'
image_input_folder = 'merged_A/images'

# 2. 원본 라벨 파일들이 들어있는 폴더
# 예: label_input_folder = 'D:/datasets/original/labels'
label_input_folder = 'merged_A/labels'

# 3. 필터링된 라벨을 "저장할" 폴더 (output)
# 예: label_output_folder = 'D:/datasets/filtered/labels'
label_output_folder = 'A_two_classes_final_2/labels'

# 4. 필터링된 이미지를 "저장할" 폴더 (output2)
# 예: image_output_folder = 'D:/datasets/filtered/images'
image_output_folder = 'A_two_classes_final_2/iamges'


# 스크립트를 실행합니다.
if __name__ == "__main__":
    paths = [image_input_folder, label_input_folder, label_output_folder, image_output_folder]
    if any("여기에" in p for p in paths):
        print("❌ 에러: 스크립트의 4개 폴더 경로 변수를 모두 실제 경로로 수정해야 합니다.")
    else:
        filter_yolo_data_and_images(
            image_input_folder,
            label_input_folder,
            label_output_folder,
            image_output_folder
        )