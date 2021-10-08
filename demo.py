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
from maybe_utils import *

# 변환행렬 계산

# 수경 기준
# 눈안쪽, 입술끝
rgb = np.array([[223, 225, 327, 332],  # 눈 좌,우 입 좌,우, 포토샵에서 세로, y
                [350, 406, 341, 409],  # 포토샵에서 가로,x
                [223 * 350, 225 * 406, 327 * 341, 332 * 409],
                [1, 1, 1, 1]])
ir = np.array([[210, 212, 314, 319],  # 눈 좌,우 입 좌,우 포토샵에서 세로, y
               [379, 434, 371, 438],  # 포토샵에서 가로,x
               [0, 0, 0, 0],
               [0, 0, 0, 0]])

#### 영상 1/4 할 때
diag = np.diag(np.array([1 / 2, 1 / 2, 1 / 4, 1]))  # 행렬곱 위한 diagonal matrix
rgb_q = np.dot(diag, rgb) #rgb quarter
ir_q = ir / 2 #ir quarter

inv_rgb = np.linalg.inv(rgb_q)
T = np.dot(ir_q, inv_rgb)  # (4, 4)

#### 원본 영상에서 사용하는 용
inv_rgb2=np.linalg.inv(rgb) #역행렬
T2=np.dot(ir,inv_rgb2)

def changexy(list):
    # list: 변환행렬 T가 곱해져서 [y,x,0,0]형태인 1차원 배열
    return [list[1],list[0]]

def flip1d(list,w, h): # 처음 변환행렬 곱하기 전 1차원 행렬
    #[[y],[x],[y*x],[1]] 형태
    #width=640
    #height=480
    x=list[1][0]
    y=list[0][0]
    return [[y],[w-x],[(y)*(w-x)],[1]]

def flip1d2(list,w,h): # T곱한 후의 1차원 행렬
    #[y,x,0,0] 형태
    #width=640
    #height=480
    x=list[1]
    y=list[0]
    return [y,w-x,0,0]

def run(vstream, vstream2, detector, rppg, time_length, horizontal_flip=False, output_path=None):
    while True:
        # 프레임 읽어오기
        ok, frame = vstream.read()
        ok_ir, frame_ir = vstream2.read()
        if not ok:
            break
        if not ok_ir:
            break

        # 좌우 반전(optional)
        if horizontal_flip:
            frame = cv2.flip(frame, 1)  # 원본영상 뒤집음
            frame_ir = cv2.flip(frame_ir, 1)  # ir영상 뒤집음

        # 1) 얼굴 영역 검출 (가장 큰 얼굴)
        bbox = detector.detect_face(frame)  # (x,y,w,h)순

        if bbox:
            # 2) 피부 마스크 생성
            face = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]  # tracking한 얼굴 영역, frame은 전체 영상
            mask = create_skin_mask(face)  # 피부영역 건하ver / 원본 코드

            # 3) 얼굴 피부 픽셀 기반 rPPG 계산
            value = rppg.extract_rppg(face, mask)

        else:
            # 얼굴이 검출되지 않으면 이전 값 또는 0으로 대체

            value = rppg.buffer[-1] if rppg.curr_buffer_size > 0 else 0
            bbox = [0, 0, 0, 0]

        # buffer 업데이트
        rppg.update_signal(value)

        # 검출된 얼굴 박스 그리기
        if bbox:
            if rppg.curr_buffer_size >= rppg.window_length:
                box_color = COLOR_GREEN
            else:
                box_color = COLOR_RED
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
            # cv2.rectangle(frame_ir, (bbox2[0], bbox2[1]), (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]), box_color, 2)

        '''
        수정된 과정
        1) rgb 채널을 각각 정규화
        2) 밝기 이진화 
        3) bbox 안쪽만 남기기 (bbox 바깥은 검정색 0으로 채우기)
        4) 모폴로지
        5) 라벨링 -> 이미지 축소
        6) 캘리브레이션 : ir영상에서의 피부영역 픽셀 구하기 -> 이미지 확대(원상복구)
        7) ir영상에 피부영역 표시하기
        8) ir영상에서 밝기값 뽑기
        '''
        # ------------------------------------------------------------------------
        # 1) 피부색 mask 생성 후 이진화

        #HSV+YCbCr로 test
        #binary = get_skin_mask_with_HSV(frame)
        #cv2.imshow('HSV1', binary)  # 수치 조정 필요할듯

        #YCbCr로 test
        #binary = get_skin_mask_with_YCrCb(frame)
        #cv2.imshow('Ycbcr1', binary)
        binary=get_skin_mask_with_YCrCb(frame)

        #RGB결합
        #get_skin_mask_with_BGR(frame)

        #조명정규화 확인용
        #LAB_lightening_norm(frame)
        # ------------------------------------------------------------------------
        # 2) bbox 바깥만 0으로 채우기
        # bbox안쪽은 그대로, 바깥만 0으로 채우기
        binary[0:bbox[1], 0:] = 0  # numpy.ndarray
        binary[bbox[1] + bbox[3]:, 0:] = 0
        binary[bbox[1]:bbox[1] + bbox[3], 0:bbox[0]] = 0
        binary[bbox[1]:bbox[1] + bbox[3], bbox[0] + bbox[2]:] = 0

        # cv2.imshow('bin', binary)
        if horizontal_flip:
            binary = cv2.flip(binary, 1)  # 이진영상 뒤집음

        # ------------------------------------------------------------------------
        # 3) 모폴로지 연산 : 팽창 -> 침식 -> 침식
        kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilation = cv2.dilate(binary, kernel_2)
        erosion = cv2.erode(dilation, kernel_5, 2)  # (480, 640)

        # ------------------------------------------------------------------------
        # 4) 라벨링
        '''
        cnt : 객체수+1(배경 포함)
        labels : 객체에 번호가 지정된 레이블 맵
        stats : connected components를 감싸는 bbox와 픽셀 정보를 가짐
        centroids : 각 connected components의 무게중심 위치
        '''
        _, src_bin = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU)  # otsu임계값 결정방법 사용
        cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
        dst1 = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)  # 모든 물체 찾기
        dst2 = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)  # 가장 큰 물체만 남긴 결과
        # print("객체수+1 : ", cnt)

        area_dict = {}
        for i in range(1, cnt):  # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
            (x, y, w, h, area) = stats[i]
            area_dict[i] = area

            # 노이즈 제거
            if area < 20:
                continue
            # 찾은 모든 blob에 박스 그리기
            cv2.rectangle(dst1, (x, y, w, h), (0, 255, 255))

        # 가장 큰 blob에 박스 그리기

        if bbox != [0, 0, 0, 0]:  # 영점설정한 것이 아니면
            max_key = max(area_dict, key=area_dict.get)
            max_value = max(area_dict.values())

            (x, y, w, h, area) = stats[max_key]
            cv2.rectangle(dst1, (x, y, w, h), (0, 0, 255))

            # cv2.imshow('dst_1', dst1)

            # 가장 큰 blob을 제외한 영역은 배경으로 만들기
            for i in range(1, cnt):
                (x, y, w, h, area) = stats[i]

                # if (20 < area and area < max_value ):
                if (area < max_value):
                    # print("index, x, y, w, h, area: ", i, x, y, w, h, area)
                    dst2[y:y + h, x:x + w] = (0, 0, 0)  # (x, y) 반대로 찍힘

            (x, y, w, h, area) = stats[max_key]
            cv2.rectangle(dst2, (x, y, w, h), (0, 0, 255))

        # 1/4로 축소
        dst2_shrink = cv2.resize(dst2, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # (240, 320, 3), 완성

        # cv2.imshow('dst_2', dst2) #rgb에서 얼굴영역만 흰색으로 해서 출력
        #cv2.imshow('shrink', dst2_shrink)  # dst 1/4

        # ------------------------------------------------------------------------
        # 5) 캘리브레이션
        # dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
        dst2 = cv2.cvtColor(dst2_shrink, cv2.COLOR_BGR2GRAY)  # (240, 320)
        height = dst2.shape[0]  # dst2행렬의 세로 길이, 240
        width = dst2.shape[1]  # dst2행렬의 가로 길이, 320

        # ir_array = np.zeros((480, 640))  # ir영상에서 피부영역 좌표값
        ir_array = np.zeros((height, width))  # ir영상에서 피부영역 좌표값
        for i in range(0, height):
            for j in range(0, width):
                if (dst2[i, j] == 255):  # rgb영상에서 피부영역이면
                    rgb_skin = np.array([[i], [j], [i * j], [1]])
                    global ir_skin
                    ir_skin = np.dot(T, rgb_skin)  # 일단 ir영상에서의 피부영역 좌표까지는 얻음
                    ir_array[int(ir_skin[0]), int(ir_skin[1])] = 1  # 피부영역만 1로 채우기

        # 좌우 반전(optional)
        if horizontal_flip:
            ir_array = cv2.flip(ir_array, 1)

        # 닫힘(팽창 ->침식)연산으로 물체의 작은 검은색 구멍 메꾸기
        ir_array = cv2.dilate(ir_array, kernel_2)
        ir_array = cv2.erode(ir_array, kernel_2)
        # cv2.imshow('small', ir_array)  # (240, 320)

        # ir_array = cv2.resize(ir_array, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)  # 뿌옇게 나옴
        ir_array = cv2.resize(ir_array, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
        # cv2.imshow('big', ir_array)

        # ------------------------------------------------------------------------
        # 6) ir영상에 피부영역 표시하기
        ir_gray = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)

        # clipping
        for w in range(640):
            for h in range(480):
                if ir_array[h][w] == 1:
                    if ir_gray[h][w] + 50 > 255:
                        ir_gray[h][w] = 255
                    else:
                        ir_gray[h][w] += 50

        cv2.imshow("ir_gray", ir_gray)
        print('gtype',type(ir_gray))
        # 결과 시각화
        cv2.imshow('original', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        #
        # frame_counter += 1

        # ---------------------------------------------
        # 7) 16x16 나눠서 밝기값 검출하기
        ## 7-1) ir영상에서 검출된 영역 찾기
        bbox2 = detector.detect_face(frame_ir)
        print(bbox2)
        #print('bbox:',bbox) #bbox순서: x,y, x차, y차
        if not bbox2: #bbox2가 자동으로 찾아지지 않는 경우
            print('can not find bbox2 in frame_ir automatically')
            bbox_lt=[[bbox[1]],[bbox[0]],[bbox[1]*bbox[0]],[1]] #왼쪽 상단, y,x순서
            bbox_lb=[[bbox[1]+bbox[3]],[bbox[0]],[bbox[0]*(bbox[1]+bbox[3])],[1]] #왼쪽 하단, y,x순서
            bbox_rt=[[bbox[1]],[bbox[0]+bbox[2]],[bbox[1]*(bbox[0]+bbox[2])],[1]] #오른쪽 상단, y,x순서
            bbox_rb=[[bbox[1]+bbox[3]],[bbox[0]+bbox[2]],[(bbox[1]+bbox[3])*(bbox[0]+bbox[2])],[1]] #오른쪽 하단, y,x순서

            if horizontal_flip: # T2가 horizontal_flip=False에 맞춰져 있어서 다시 계산 -> x뒤집은 후 다시 뒤집어야함
                bbox_lt=flip1d(bbox_lt,width*2,height*2)
                bbox_lb=flip1d(bbox_lb,width*2,height*2)
                bbox_rt=flip1d(bbox_rt,width*2,height*2)
                bbox_rb=flip1d(bbox_rb,width*2,height*2)

            bbox_lt2=np.dot(T2, bbox_lt).ravel()
            bbox_lb2=np.dot(T2, bbox_lb).ravel()
            bbox_rt2=np.dot(T2, bbox_rt).ravel()
            bbox_rb2=np.dot(T2, bbox_rb).ravel()

            if horizontal_flip: # 다시 x 뒤집어줌
                bbox_lt2=flip1d2(bbox_lt2,width*2,height*2)
                bbox_lb2=flip1d2(bbox_lb2,width*2,height*2)
                bbox_rt2=flip1d2(bbox_rt2,width*2,height*2)
                bbox_rb2=flip1d2(bbox_rb2,width*2,height*2)
            #print(bbox_lt2, bbox_lb2, bbox_rt2, bbox_rb2)


            # 큰 좌표 기준으로 맞춰줌
            ## 좌표 비교, [y,x,0,0] 형태로 들어있음!
            if bbox_lt2[0]>=bbox_rt2[0]: # 좌측 상단 y값->작은것 pick
                bbox_ly=bbox_rt2[0]
            else:
                bbox_ly=bbox_lt2[0]
            if bbox_lt2[1]>=bbox_lb2[1]: #좌측 상단 x값->작은것 pick
                bbox_lx=bbox_lb2[1]
            else:
                bbox_lx=bbox_lt2[1]
            if bbox_lb2[0]>=bbox_rb2[0]: #우측 하단 y값->큰것 pick
                bbox_ry=bbox_lb2[0]
            else:
                bbox_ry=bbox_rb2[0]
            if bbox_rt2[1]>=bbox_rb2[1]:#우측 하단 x값->큰 것 pick
                bbox_rx=bbox_rt2[1]
            else:
                bbox_rx=bbox_rb2[1]

            bbox_l=[bbox_ly,bbox_lx,0,0]
            bbox_r=[bbox_ry,bbox_rx,0,0]
            #print(bbox_l,bbox_r)

            #x,y순서로 바꿔줌
            bbox_l=changexy(bbox_l)
            bbox_r=changexy(bbox_r)
            bbox2 = [int(bbox_l[0]), int(bbox_l[1]), int(bbox_r[0])-int(bbox_l[0]), int(bbox_r[1])-int(bbox_l[1])] #x, y, x차, y차
            #print("bbox2: ", bbox2)


        #확인용
        cv2.rectangle(ir_gray, (bbox2[0], bbox2[1]), (bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]), box_color, 2)
        #cv2.imshow("ir_gray_box", ir_gray) # ir영상에 box 그린 것
        ir_gray2 = ir_gray[bbox2[1]:bbox2[1] + bbox2[3], bbox2[0]:bbox2[0] + bbox2[2]] #crop한 영역 확인
        cv2.imshow("ir_gray2", ir_gray2)

        ## 7-2) 16x16나눠서 밝기값 구하기
        num=16
        box_height = ir_gray2.shape[0]  #bbox2의 세로
        box_width = ir_gray2.shape[1]  #bbox2의 가로
        distan_w = box_width // num  #
        distan_h = box_height // num  # #bbox2를 16등분한 것, 그거의 width

        light_1d = []
        ir_num=[[0 for i in range(num)] for j in range(num)]#피부마스크=1인 영역의 개수를 세기 위함, 16x16으로 해서 4개씩 더해서 평균을 내는 용도

        # 피부 mask=1인 영역의 개수 세기
        ir_mask=ir_array[bbox2[1]:bbox2[1] + bbox2[3], bbox2[0]:bbox2[0] + bbox2[2]] #ir maxk 배열(640*480)에서 피부영역만 잘라내기
        cv2.imshow("ir_mask", ir_mask)

        for i in range(num):#세로 h
            for j in range(num):#가로 w
                # bbox2 구간에서 mask만 있는 배열을 2차원 배열로 16x16영역 사각형(1개) 자르기
                if i==num-1 and j==num-1: #맨구석끝
                    divide_irmask = ir_mask[i * distan_h:, j * distan_w:]
                elif i==num-1: #아래 끝
                    divide_irmask = ir_mask[i * distan_h:, j * distan_w:(j + 1) * distan_w]
                elif j==num-1: #오른쪽 끝
                    divide_irmask = ir_mask[i * distan_h:(i + 1) * distan_h, j * distan_w:]
                else:
                    divide_irmask=ir_mask[i*distan_h:(i+1)*distan_h,j*distan_w:(j+1)*distan_w]

                # 2차원 배열 안의 원소들 1차원으로 펴기
                divide_irmask=np.array(divide_irmask).flatten().tolist()

                # 1차원 배열에서 1 개수 count
                ir_num[i][j]=divide_irmask.count(1)

        # 피부 영역 1로 표시하기
        ir_skin = [[0 for i in range(640)] for j in range(480)] #피부마스크=1인 영역에 대하여 ir 피부 화소값 가지고 있는 배열
        ir_ = cv2.cvtColor(frame_ir, cv2.COLOR_BGR2GRAY)#ir_gray는 ir+피부마스크, frame_ir은 3채널임(BGR) -> GRAY로변환
        for w in range(640):
            for h in range(480):
                if ir_array[h][w] == 1: #피부 mask 1인 영역에 대하여
                        ir_skin[h][w] = ir_[h][w]

        ir_skin = np.array(ir_skin) #480,640

        for m in range(num - 1): #h
            for n in range(num - 1): #w
                if m==num-2 and n==num-2:
                    lightness = ir_skin[bbox2[1]+m * distan_h:bbox2[1]+bbox2[3],bbox2[0]+n * distan_w:bbox2[0]+bbox2[2]] #lightness: ir영상(밝기)에서 4개 선택한 것, ir skin 640 480임에 유의
                elif m==num-2:
                    lightness = ir_skin[bbox2[1]+m * distan_h:bbox2[1]+bbox2[3],bbox2[0]+n * distan_w: bbox2[0]+(n + 2) * distan_w]
                elif n==num-2:
                    lightness = ir_skin[bbox2[1]+m * distan_h: bbox2[1]+(m + 2) * distan_h,bbox2[0]+n * distan_w:bbox2[0]+bbox2[2]]
                else:
                    lightness = ir_skin[bbox2[1]+m * distan_h: bbox2[1]+(m + 2) * distan_h,bbox2[0]+n * distan_w: bbox2[0]+(n + 2) * distan_w]

                ir_num_sum=ir_num[m][n]+ir_num[m+1][n]+ir_num[m][n+1]+ir_num[m+1][n+1]#16x16에서 묶은 4개에 대한 개수 합 구함


                if ir_num_sum: #0 아니면
                    lightness=np.array(lightness).flatten().tolist()
                    lightness = sum(lightness)/ir_num_sum #피부 영역에 대한 것만 밝기 합/피부 영역 화소 개수
                else:
                    lightness=0.0
                light_1d.append(lightness)

        print('light 1d\n',light_1d)

        light_2d = np.reshape(np.array(light_1d), (15, 15))#엑셀 저장할거면 1차원으로 하는게 더 좋을것같으나 시각화용..
        #print(light_2d)
        light_2d=light_2d/255 #0-1사이로 나타내야하는듯..?ㅠㅠ
        cv2.imshow('light_small', light_2d)
        #light_2d = cv2.resize(light_2d, None, fx=8, fy=8,interpolation=None) #보간법 없앨순없을까ㅠㅠ
        #cv2.imshow('light', light_2d)

        print("-----------------")


    # 리소스 해제
    cv2.destroyAllWindows()
    vstream.close()
    vstream2.close()
    # if not isinstance(output_path, type(None)):
    #     fout.close()


if __name__ == '__main__':
    # 비디오 관리 객체 생성 - 원본 코드
    # vstream = VideoStream(0)  # 카메라
    # vstream = VideoStream('../data/videos/우혁/RGB.avi')
    # vstream2 = VideoStream('../data/videos/우혁/IR.mp4')

    #vstream = VideoStream('../data/실험1_공포영상_중간중간/videos/나연/14_16-40-50RGB.mp4')
    #vstream2 = VideoStream('../data/실험1_공포영상_중간중간/videos/나연/14_16-40-50IR.mp4')
    #vstream = VideoStream('../data/실험1_공포영상_중간중간/videos/소현/14_17-48-36RGB.mp4')
    #vstream2 = VideoStream('../data/실험1_공포영상_중간중간/videos/소현/14_17-48-36IR.mp4')
    #vstream = VideoStream('../data/videos/지원/14_17-05-14RGB.mp4')
    #vstream2 = VideoStream('../data/videos/지원/14_17-05-14IR.mp4')
    #vstream = VideoStream('../data/videos/나혜/21_17-48-10RGB.mp4')
    #vstream2 = VideoStream('../data/videos/나혜/21_17-48-10IR.mp4')
    vstream = VideoStream('../data/videos/수경/21_18-14-36RGB.mp4')
    vstream2 = VideoStream('../data/videos/수경/21_18-14-36IR.mp4')

    w = vstream.width
    h = vstream.height
    fps = 30

    # 얼굴 검출기 생성
    detector = FaceDetector(model_path='./model/opencv_face_detector_uint8.pb',
                            config_path='./model/opencv_face_detector.pbtxt')

    # Remote-PPG 추출 객체 생성
    FPS = 30  # 입력 비디오의 프레임율
    DURATION = 8  # 신호 처리를 위한 타임 윈도우의 길이(단위: 초)
    PASS_BAND = (42, 240)  # 신호 필터링을 위한 통과 대역(단위: bpm)
    rppg = RemotePPG(FPS, FPS * DURATION, PASS_BAND)

    # 심박 측정 수행
    run(vstream, vstream2, detector, rppg,
        time_length=FPS * 6,
        horizontal_flip=True,
        output_path='./output.csv')