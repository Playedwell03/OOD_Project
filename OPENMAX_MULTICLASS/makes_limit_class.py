import os

def filter_labels(input_dir, output_dir, allowed_classes={0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        with open(input_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and parts[0].isdigit() and int(parts[0]) in allowed_classes:
                filtered_lines.append(line)

        # 하나도 남지 않으면 파일 생성하지 않음
        if filtered_lines:
            with open(output_path, 'w') as f:
                f.writelines(filtered_lines)

    print(f"Done. Filtered labels saved in: {output_dir}")
    
filter_labels('merged_data/labels', 'one_labels_data_v1/labels')