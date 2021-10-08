"""
OpenCV DNN face detector
"""

import cv2
import numpy as np
from src.tracker import KCFTracker
from src.config import *


class FaceDetector:
    def __init__(self, model_path, config_path, track_toler=1):
        try:
            self.detector = cv2.dnn.readNetFromTensorflow(model_path, config_path)  # 학습된 모델 불러오기
        except FileNotFoundError:
            raise Exception("[ERROR] Invalid file path. Check 'model_path' or 'config_path'")

        self.is_tracking = False        # 추적 유무
        self.tracker = KCFTracker()     # 객체 추적기
        self.track_toler = track_toler  # 허용 오차(단위: 픽셀)
        self.prev_rect = [0, 0, 0, 0]   # 이전 얼굴 영역

    def detect_face(self, frame, confidence=0.6):
        rect = []               # 중간 얼굴 영역 결과 저장할 리스트
        curr_rect = []          # 현재 얼굴 영역 저장할 리스트
        h, w = frame.shape[:2]  # 영상의 height, width 정보

        # 현재 추적 중인 얼굴이 존재하지 않으면, 얼굴 검출 수행
        if not self.is_tracking:
            # 얼굴 검출 수행
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), [104., 117., 123.], False, True)  # 이미지를 4차원의 blob(N차원 배열)으로 변환
            self.detector.setInput(blob)
            detections = self.detector.forward()

            # 설정한 confidence보다 높은 얼굴 영역만 남김
            rects = [detections[0, 0, i, 3:7] for i in range(detections.shape[2]) if detections[0, 0, i, 2] >= confidence]
            if len(rects) > 0:
                # 가장 큰 하나의 얼굴만 남김
                rects = sorted(rects, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
                rect = (rects[0] * np.array([w, h, w, h])).astype('int')
                rect = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])  # (xs,ys,xe,ye) -> (x,y,w,h)

                # 얼굴 크기가 유효할 경우 추적기에 등록
                self.tracker.init(frame, rect)  # (x,y,w,h)
                self.is_tracking = True

        # 현재 추적 중인 얼굴이 존재하면, 추적기로부터 추적된 위치 업데이트
        else:
            self.is_tracking, rect = self.tracker.update(frame)
            rect = [int(r) for r in rect]

        # 얼굴 영역 유효성 검사
        if self.is_tracking and rect[0] >= 0 and rect[0] + rect[2] < w and rect[1] >= 0 and rect[1] + rect[3] < h and \
                rect[2] >= MIN_FACE_SIZE and rect[3] >= MIN_FACE_SIZE:
            # 얼굴의 위치 및 크기 변화량이 특정값보다 작으면 이전 위치로 사용(얼굴 영역 떨림을 보완하기 위함)
            curr_rect = [curr if abs(curr - prev) > self.track_toler else prev for curr, prev in zip(rect, self.prev_rect)]
            self.prev_rect = curr_rect[:]
        else:
            self.is_tracking = False

        return curr_rect
