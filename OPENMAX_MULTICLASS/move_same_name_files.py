import os
import shutil

def copy_matching_files(
    dir1_images, dir1_labels,
    dir2_images, dir2_labels,
    output_dir_images, output_dir_labels
):
    os.makedirs(output_dir_images, exist_ok=True)
    os.makedirs(output_dir_labels, exist_ok=True)

    # dir2의 파일 이름 집합 (확장자 제외)
    label_basenames = {os.path.splitext(f)[0] for f in os.listdir(dir2_labels)}
    image_basenames = {os.path.splitext(f)[0] for f in os.listdir(dir2_images)}
    common_basenames = label_basenames & image_basenames

    print(f"Matching file count: {len(common_basenames)}")

    for base in common_basenames:
        img_file = base + ".jpg"  # 필요시 확장자 변경 가능
        label_file = base + ".txt"

        src_img = os.path.join(dir1_images, img_file)
        src_lbl = os.path.join(dir1_labels, label_file)

        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, os.path.join(output_dir_images, img_file))
            shutil.copy2(src_lbl, os.path.join(output_dir_labels, label_file))
        else:
            print(f"Warning: File missing in dir1 for {base}")

    print("Copy complete.")

# 예시 사용법
copy_matching_files(
    dir1_images="merged_data/images",
    dir1_labels="merged_data/labels",
    dir2_images="trash_removed/all_removed/images",
    dir2_labels="trash_removed/all_removed/labels",
    output_dir_images="clean/images",
    output_dir_labels="clean/labels"
)