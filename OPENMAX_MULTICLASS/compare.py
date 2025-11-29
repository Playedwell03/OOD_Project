import os

def compare_images(normal_dir, target_dir, trash_dir):
    def get_image_names(directory):
        return set(os.path.splitext(f)[0] for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    normal_set = get_image_names(normal_dir)
    target_set = get_image_names(target_dir)
    trash_set = get_image_names(trash_dir)

    normal_overlap = target_set & normal_set
    trash_overlap = target_set & trash_set
    others = target_set - (normal_set | trash_set)

    print("비교 결과:")
    print(f"전체 데이터 개수: {len(normal_overlap) + len(trash_overlap)}")
    print(f"정상 데이터 포함: {len(normal_overlap)}개")
    print(f"쓰레기 데이터 포함: {len(trash_overlap)}개")
    

    return {
        "normal": list(normal_overlap),
        "trash": list(trash_overlap),
        "others": list(others)
    }

# 사용 예시
result = compare_images(
    normal_dir='merged_data/images',    # 정상 데이터 폴더
    target_dir='final_data_origin_v2/images',        # 비교 대상 폴더
    trash_dir='multiclass_fish/images'       # 쓰레기 데이터 폴더
)