import cv2
import numpy as np
from src.config import *


def create_skin_mask(img):
    """ 입력 영상에 대한 피부색 마스크 생성

    :param img: 대상 입력 영상
    :return: 피부색 마스크 (1: skin pixel / 0: non-skin pixel)
    """
    # 유효 피부색 범위
    low = np.array([0, 133, 77], np.uint8)
    high = np.array([235, 173, 127], np.uint8)

    # YCbCr 색공간 기반 피부색 필터링
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, low, high)
    mask[mask == 255] = 1
    return mask


def draw_graph(arr, heartrate, width):
    """ 결과신호 및 심박수 시각화를 위한 함수

    :param arr: 1차원 시계열 신호
    :param heartrate: 심박수
    :param width: 영상의 너비 정보
    :return: 정보가 시각화된 2차원 그래프 배열
    """
    graph = np.zeros((width//4, width, 3), dtype='uint8')
    length = len(arr)

    try:
        if length >= 3:
            # 그래프 위치 관련 변수
            offset_y = int(graph.shape[0] * 0.05)
            offset_x = 160

            xs = 0
            xe = graph.shape[1] - offset_x
            ys = offset_y
            ye = graph.shape[0] - offset_y
            stride = (xe - xs) / len(arr)

            # 신호를 0.0 ~ 1.0로 정규화
            normalized = cv2.normalize(np.array(arr), None, ys, ye, cv2.NORM_MINMAX).ravel()

            # 그래프 그리기
            for i in range(length - 2):
                _x1 = xs + int(i * stride)
                _y1 = int(normalized[i])
                _x2 = xs + int((i + 1) * stride)
                _y2 = int(normalized[i + 1])
                cv2.line(graph, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)

            # 마지막 열 그리기
            _x1 = xs + int((length - 2) * stride)
            _y1 = int(normalized[-2])
            _x2 = xe
            _y2 = int(normalized[-1])
            cv2.line(graph, (_x1, _y1), (_x2, _y2), (0, 255, 0), 2)

            # 그래프 우측에 심박수 값 시각화
            cv2.putText(graph, '%3d' % int(heartrate), (width - 150, ye - 60), FONT, 2.5, COLOR_RED, 4)
            cv2.putText(graph, 'bpm', (width - 80, ye - 15), FONT, 1.0, COLOR_RED, 2)
    except ValueError:
        pass

    return graph
