OOD_Project

echo "# OOD_Project" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Playedwell03/OOD_Project.git
git push -u origin main

# 프로젝트 진행상황(12.03)
기존 프로젝트의 목적은 "쓰레기 데이터들을 걸러 모델 성능 향상에 기여하자" 였음.

쓰레기? 쓰레기란 무엇인가?에 대하여 생각함.
쓰레기 데이터를 "학습에 방해가 되는 데이터"라고 정의한 뒤, 다음과 같이 세 가지로 분류.
1. 기존 데이터셋과 주제가 전혀 다른 이미지의 데이터셋
2. 바운딩 박스 정보가 잘못 된 데이터
3. 클래스 정보가 잘못 된 데이터

그렇다면 정상 데이터는 무엇인가?
[ROBOFLOW의 traffic sign 데이터셋]
https://universe.roboflow.com/project-fatj9/traffic_sign-fa883/dataset/8

img_size : 416 x 416
데이터 개수 : 4,056(라벨 기준)

위의 데이터셋에서 쓰레기 데이터가 아닌 것이 정상 데이터가 되겠음.

이후 정의한 각각의 쓰레기 데이터들을 기존 데이터셋을 이용해 만든 뒤, 학습 및 평가 진행함.

