import os

def convert_yolo_labels_to_single_class(input_dir, output_dir, target_classes, new_class_id):
    """
    YOLO 형식의 라벨 파일에서 지정된 클래스들을 새로운 클래스 ID로 변경합니다.

    Args:
        input_dir (str): 원본 라벨 파일들이 있는 폴더 경로
        output_dir (str): 변경된 라벨 파일들을 저장할 폴더 경로
        target_classes (list): 변경할 대상 클래스 ID 리스트 (예: list(range(1, 16)))
        new_class_id (int): 새로 지정할 클래스 ID (예: 0)
    """
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 입력 폴더의 모든 파일 순회
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
                for line in infile:
                    # 각 줄을 공백 기준으로 분리
                    parts = line.strip().split()
                    
                    if parts:
                        try:
                            # 클래스 ID를 정수로 변환
                            class_id = int(parts[0])
                            
                            # 클래스 ID가 변경 대상에 포함되는지 확인
                            if class_id in target_classes:
                                parts[0] = str(new_class_id)
                                new_line = ' '.join(parts) + '\n'
                                outfile.write(new_line)
                            else:
                                # 그 외의 경우는 그대로 파일에 씀
                                outfile.write(line)
                        except (ValueError, IndexError):
                            # 클래스 ID가 숫자가 아니거나 줄이 비어있는 등 예외 발생 시 원본 라인 유지
                            outfile.write(line)

    print(f"라벨 변환이 완료되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")


# --- 사용 예시 ---
# 아래 경로와 변수를 실제 환경에 맞게 수정하여 사용하세요.

# 원본 라벨 파일들이 있는 폴더
input_folder = "multiclass_fish/labels" 

# 변환된 라벨 파일들을 저장할 폴더
output_folder = "fish_class_0/labels" 

# 변경할 클래스 ID 리스트 (1부터 15까지)
target_classes_to_change = list(range(1, 16))

# 새로 지정할 클래스 ID
new_class = 0

# 함수 호출
convert_yolo_labels_to_single_class(input_folder, output_folder, target_classes_to_change, new_class)