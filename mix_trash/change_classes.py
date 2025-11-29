import os

def convert_yolo_labels(input_dir, output_dir):
    """
    YOLOv5 형식의 라벨 파일에서 클래스 5를 0으로 변경합니다.

    Args:
        input_dir (str): 원본 라벨 파일들이 있는 폴더 경로
        output_dir (str): 변경된 라벨 파일들을 저장할 폴더 경로
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
                    
                    # 클래스 ID가 '5'이면 '0'으로 변경
                    if parts and parts[0] == '5':
                        parts[0] = '0'
                        new_line = ' '.join(parts) + '\n'
                        outfile.write(new_line)
                    else:
                        # 그 외의 경우는 그대로 파일에 씀
                        outfile.write(line)

    print(f"라벨 변환이 완료되었습니다. 결과는 '{output_dir}' 폴더에 저장되었습니다.")


# --- 사용 예시 ---
# 아래 'input_folder'와 'output_folder' 경로를 실제 경로로 수정하여 사용하세요.

# 원본 라벨 파일들이 있는 폴더
input_folder = "A_two_classes_final_2/labels2" 

# 변환된 라벨 파일들을 저장할 폴더
output_folder = "A_two_classes_final_2/labels" 

# 함수 호출
convert_yolo_labels(input_folder, output_folder)