import cv2
import datetime

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record = False


# 카메라를 추가적으로 연결하여 외장 카메라를 이용하는 경우 장치 번호가 1~n 까지 순차적으로 할당
rgb_capture = cv2.VideoCapture(1) #cv2.CAP_DSHOW 쓰면 열리는 속도는 빨라지지만 영상이 왜곡생기고 느려짐
ir_capture  = cv2.VideoCapture(2) # 카메라의 장치 번호(ID)와 연결한다. Index는 카메라의 장치 번호를 의미한다.
print('카메라 불러옴')

ir_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
ir_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rgb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
rgb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print('frame set완료')

while True:
    print('---')
    ret, frame = ir_capture.read()
    ret2, frame2 = rgb_capture.read()

    cv2.imshow("ir_vid", frame)
    cv2.imshow('rgb_vid', frame2)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 24:
        print('녹화 시작')
        print('frame',cv2.CAP_PROP_FPS)
        record = True
        ir_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "IR.mp4", fourcc, cv2.CAP_PROP_FPS, (frame.shape[1], frame.shape[0]))
        rgb_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "RGB.mp4", fourcc,cv2.CAP_PROP_FPS, (frame2.shape[1], frame2.shape[0]))

    elif key == 3:
        print("녹화 중지")
        record = False
        break

    if record==True:
        print("녹화중")
        ir_video.write(frame)
        rgb_video.write(frame2)

ir_capture.release()
rgb_capture.release()

cv2.destroyAllWindows()

# =============================

import cv2
import datetime

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record = False


# 카메라를 추가적으로 연결하여 외장 카메라를 이용하는 경우 장치 번호가 1~n 까지 순차적으로 할당
rgb_capture = cv2.VideoCapture(1) #cv2.CAP_DSHOW 쓰면 열리는 속도는 빨라지지만 영상이 왜곡생기고 느려짐
ir_capture  = cv2.VideoCapture(2) # 카메라의 장치 번호(ID)와 연결한다. Index는 카메라의 장치 번호를 의미한다.
print('카메라 불러옴')

ir_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
ir_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rgb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
rgb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print('frame set완료')

while True:
    print('---')
    ret, frame = ir_capture.read()
    ret2, frame2 = rgb_capture.read()

    cv2.imshow("ir_vid", frame)
    cv2.imshow('rgb_vid', frame2)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 24:
        print('녹화 시작')
        print('frame',cv2.CAP_PROP_FPS)
        record = True
        ir_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "IR.mp4", fourcc, cv2.CAP_PROP_FPS, (frame.shape[1], frame.shape[0]))
        rgb_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "RGB.mp4", fourcc,cv2.CAP_PROP_FPS, (frame2.shape[1], frame2.shape[0]))

    elif key == 3:
        print("녹화 중지")
        record = False
        break

    if record==True:
        print("녹화중")
        ir_video.write(frame)
        rgb_video.write(frame2)

ir_capture.release()
rgb_capture.release()

cv2.destroyAllWindows()

# =====================================================

import cv2
import datetime

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
record = False


# 카메라를 추가적으로 연결하여 외장 카메라를 이용하는 경우 장치 번호가 1~n 까지 순차적으로 할당
rgb_capture = cv2.VideoCapture(1) #cv2.CAP_DSHOW 쓰면 열리는 속도는 빨라지지만 영상이 왜곡생기고 느려짐
ir_capture  = cv2.VideoCapture(2) # 카메라의 장치 번호(ID)와 연결한다. Index는 카메라의 장치 번호를 의미한다.
print('카메라 불러옴')

ir_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
ir_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
rgb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
rgb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print('frame set완료')

while True:
    print('---')
    ret, frame = ir_capture.read()
    ret2, frame2 = rgb_capture.read()

    cv2.imshow("ir_vid", frame)
    cv2.imshow('rgb_vid', frame2)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)

    if key == 24:
        print('녹화 시작')
        print('frame',cv2.CAP_PROP_FPS)
        record = True
        ir_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "IR.mp4", fourcc, cv2.CAP_PROP_FPS, (frame.shape[1], frame.shape[0]))
        rgb_video = cv2.VideoWriter("C:/Users/minji/Desktop/" + str(now) + "RGB.mp4", fourcc,cv2.CAP_PROP_FPS, (frame2.shape[1], frame2.shape[0]))

    elif key == 3:
        print("녹화 중지")
        record = False
        break

    if record==True:
        print("녹화중")
        ir_video.write(frame)
        rgb_video.write(frame2)

ir_capture.release()
rgb_capture.release()

cv2.destroyAllWindows()