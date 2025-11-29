import os

# 디렉토리 경로 설정
dir1 = 'test_data_v3/images'
dir2 = 'data_for_learn/train/images'
dir3 = 'data_for_learn/valid/images'

# 각 디렉토리의 파일명 집합 가져오기 (서브디렉토리는 무시)
files1 = set(f for f in os.listdir(dir1) if os.path.isfile(os.path.join(dir1, f)))
files2 = set(f for f in os.listdir(dir2) if os.path.isfile(os.path.join(dir2, f)))
files3 = set(f for f in os.listdir(dir3) if os.path.isfile(os.path.join(dir3, f)))

# 겹치는 파일명 찾기
common_12 = files1 & files2
common_13 = files1 & files3
common_23 = files2 & files3
common_all = files1 & files2 & files3

# 결과 출력
if common_12:
    print("디렉토리 1과 2에서 겹치는 파일명:", common_12)
if common_13:
    print("디렉토리 1과 3에서 겹치는 파일명:", common_13)
if common_23:
    print("디렉토리 2와 3에서 겹치는 파일명:", common_23)
if common_all:
    print("모든 디렉토리에서 공통으로 겹치는 파일명:", common_all)

if not (common_12 or common_13 or common_23):
    print("세 디렉토리 간 파일명이 전혀 겹치지 않습니다.")