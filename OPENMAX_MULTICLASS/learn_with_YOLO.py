import os

# OpenMP 중복 허용
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from pathlib import Path

import torch

torch.cuda.empty_cache()  # GPU 메모리 해제


if __name__ == '__main__':
    # 1. 설정
    img_size = 640
    batch_size = 16
    epochs = 100
    weights = 'yolov5s.pt'  # yolov5s도 Ultralytics에서 사용 가능
    data_yaml = 'final_data_origin_for_learn/data.yaml'
    exp_name = 'exp_crack_signs' 
    project_dir = Path('YOLO_model/before') 

    # 2. 모델 로드
    model = YOLO(weights)  # GPU로 실행되는 기본 설정
    # model.to('cuda')  # 모델을 GPU로 전환
    model.to('cpu')  # 모델을 CPU로 전환

    # 3. 학습 수행
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name=exp_name,
        project=str(project_dir),
        exist_ok=True,
        save=True,
        save_conf=True,
        verbose=True  # 학습 로그 출력
    )


    # 4. 학습 결과 경로 확인
    exp_path = project_dir / exp_name
    print(f'\n✅ 학습 완료! 결과 저장 경로: {exp_path.resolve()}')

    # 5. 주요 결과 파일 목록 출력
    important_files = [
        'results.png',
        'confusion_matrix.png',
        'opt.yaml',
        'args.yaml',
        'weights/best.pt',
        'weights/last.pt'
    ]

    for f in important_files:
        f_path = exp_path / f
        if f_path.exists():
            print(f'📄 {f_path}')
        else:
            print(f'⚠️ {f} 누락됨')