# GSR_Callibration
remote GSR RGB카메라와 IR 카메라 캘리브레이션 방법

model : 얼굴 트래킹에 사용되는 딥러닝 기반 학습 모델

src : 트래킹, skin segmentation, videostram 등 필요한 함수 모음

demo.py : 얼굴 트래킹 -> skin segmentation -> 캘리브레이션 -> 홀 제거 -> 얼굴 영역 16x16 밝기값 추출

experiment_utils.py : 실험에 필요한 함수들

run.experiment.py : 실험 시작
