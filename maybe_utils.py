import cv2
import csv
import datetime
import time
import os
import numpy as np
from src.config import *
from src.video_stream import VideoStream
from src.face_detect import FaceDetector
from src.rppg import RemotePPG
from src.util import create_skin_mask, draw_graph
from PIL import Image
import matplotlib.pyplot as plt



# 건영오빠 코드
# ycrcb, hsv 두가지 사용해서 피부영역 검출 (https://github.com/CHEREF-Mehdi/SkinDetection)
# create_skin_mask()함수와 같은 기능이지만, create_skin_mask()함수는 ycrcb만 사용해 피부영역을 검출함
def get_skin_mask_with_HSV(frame):
    #     low = np.array([0, 133, 77], np.uint8)
    #     high = np.array([235, 173, 127], np.uint8)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(HSV, (0, 15, 0), (17, 170, 255))
    YCrCb_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    #     HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    #     YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    # merge skin detection (YCbCr and hsv)
    global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 5)
    #     global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    #     global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_CLOSE, np.ones((4,4), np.uint8))
    return global_mask

def get_skin_mask_with_YCrCb(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))  # 건영오빠 버전
    global_mask = cv2.medianBlur(YCrCb_mask, 5)
    return global_mask

# 아래의 create_skin_mask()는 imshow가 안되서 함수 고침
def get_skin_mask_with_gunha(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(ycrcb, (0, 133, 80), (235, 173, 127))  # 건하오빠 버전
    global_mask = cv2.medianBlur(YCrCb_mask, 5)
    return global_mask

# 건하오빠 피부영역 검출 코드-YCrCb 사용
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


# 조명 정규화 방법(승건오빠가 줬으나 아직 사용 안 함), 사용X
def normalized(down):
    norm = np.zeros((480, 640, 3), np.float32)  # 0으로 채워진 배열 생성
    frame = np.zeros((480, 640, 3), np.uint8)
    rgb = cv2.cvtColor(down, cv2.COLOR_RGB2BGR)  # rgb 입력이미지를 bgr로 변환

    b = rgb[:, :, 2]
    g = rgb[:, :, 1]
    r = rgb[:, :, 0]
    sum = b + g + r

    norm[:, :, 0] = b / sum * 255.0
    norm[:, :, 1] = g / sum * 255.0
    norm[:, :, 2] = r / sum * 255.0

    frame = cv2.convertScaleAbs(norm)
    return frame


# homomorphic filtering, 사용X
# bgr영상을 입력으로 받음
def filtering(img):
    ### homomorphic filter는 gray scale image에 대해서 밖에 안 되므로
    ### YUV color space로 converting한 뒤 Y에 대해 연산을 진행
    # img = cv2.imread('./imgs/ny.JPG')
    img_YUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)  # (417, 319, 3) = (H, W, 3)
    y = img_YUV[:, :, 0]  # 밝기인 y성분만 가져오기

    rows = y.shape[0]
    cols = y.shape[1]

    ### illumination elements와 reflectance elements를 분리하기 위해 log를 취함
    imgLog = np.log1p(np.array(y, dtype='float') / 255)  # y값을 0~1사이로 조정한 뒤 log(x+1)

    ### frequency를 이미지로 나타내면 4분면에 대칭적으로 나타나므로
    ### 4분면 중 하나에 이미지를 대응시키기 위해 row와 column을 2배씩 늘려줌
    M = 2 * rows + 1
    N = 2 * cols + 1

    ### gaussian mask 생성 sigma = 10
    sigma = 10
    (X, Y) = np.meshgrid(np.linspace(0, N - 1, N), np.linspace(0, M - 1, M))  # 0~N-1(and M-1) 까지 1단위로 space를 만듬
    Xc = np.ceil(N / 2)  # 올림 연산
    Yc = np.ceil(M / 2)
    gaussianNumerator = (X - Xc) ** 2 + (Y - Yc) ** 2  # 가우시안 분자 생성

    ### low pass filter와 high pass filter 생성
    LPF = np.exp(-gaussianNumerator / (2 * sigma * sigma))
    HPF = 1 - LPF

    ### LPF랑 HPF를 0이 가운데로 오도록iFFT함.
    ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음
    ### 에너지를 각 귀퉁이로 모아 줌
    LPF_shift = np.fft.ifftshift(LPF.copy())
    HPF_shift = np.fft.ifftshift(HPF.copy())

    ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔
    img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
    img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N)))  # low frequency 성분
    img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N)))  # high frequency 성분

    ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
    gamma1 = 0.3
    gamma2 = 1.5
    img_adjusting = gamma1 * img_LF[0:rows, 0:cols] + gamma2 * img_HF[0:rows, 0:cols]

    ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌
    img_exp = np.expm1(img_adjusting)  # exp(x) + 1
    img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp))  # 0~1사이로 정규화
    img_out = np.array(255 * img_exp, dtype='uint8')  # 255를 곱해서 intensity값을 만들어줌

    ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting
    img_YUV[:, :, 0] = img_out
    result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    # cv2.imshow('homomorphic', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result

