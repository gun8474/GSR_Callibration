import requests
import pandas as pd
import cv2
import pyglet
from datetime import datetime


def neulog_sensor(save_path, file_name):
    '''
    neulog api로 GSR값 측정
    코드 실행 방법 : 센서 usb를 노트북에 꽂고, neulog api를 실행시킨 후, sensor.py 코드 실행
                   실행할 때마다 센서와 neulog api를 껐다 켜야함
    출력 :GSR값, 현재시간이 담긴 csv파일
    :param save_path: neulog 데이터 저장 경로
    :param file_name: 파일명
    :return:
    '''

    # 버전
    url_version = "http://localhost:22002/NeuLogAPI?GetServerVersion"
    response_version = requests.get(url_version)
    print("\nserver version \nstatus code : ", response_version.status_code)
    print("answer : ", response_version.text)

    # 서버 상태
    url_status = "http://localhost:22002/NeuLogAPI?GetSeverStatus"
    response_status = requests.get(url_status)
    print("\nserver status \nstatus code : ", response_status.status_code)
    print("answer : ", response_status.text)

    # 센서 아이디 1로 설정
    url_id = "http://localhost:22002/NeuLogAPI?SetSensorsID:[1]"
    # params = {'param': '1'}
    response_id = requests.get(url_id)
    print("\nsensor ID \nstatus code : ", response_id.status_code)
    print("answer : ", response_id.text)

    # Start Experiment
    # 4-1000fps, 5-100fps, 6-50fps, 7-20fps, 8-10fps, 9-5fps, 10-2fps, 11-1fps
    url_start = "http://localhost:22002/NeuLogAPI?StartExperiment:[GSR],[1],[8],[10000]"  # 3번째 파라미터가 fps
    # params_start = {'param1': 'GSR', 'param2': '7'}
    response_start = requests.get(url_start)
    print("\nstart experiment \nstatus code : ", response_start.status_code)
    print("answer : ", response_start.text)

    # 데이터 읽기_중요_실질적으로 데이터 얻는 곳
    url_get = "http://localhost:22002/NeuLogAPI?GetExperimentSamples"
    while (requests.get(url_get).json()):
        value = requests.get(url_get).json()
        text = value['GetExperimentSamples']
        now = datetime.now().strftime("%H-%M-%S.%f")[:-3]
        textframe = pd.DataFrame(text[0][1:])
        textframe['time'] = now
        # textframe.to_csv("./neulog_sensor/sensor20.csv", header=False, index=False)  # csv파일로 저장
        textframe.to_csv(save_path + file_name + ".csv", header=False, index=False)  # csv파일로 저장

    # Stop Experiment
    url_stop = "http://localhost:22002/NeuLogAPI?StopExperiment"
    response_stop = requests.get(url_stop)
    print("\nstop experiment \nstatus code : ", response_stop.status_code)
    print("answer : ", response_stop.text)


def show_video(vid_path, w, h):
    '''
    시각 자극영상 재생(영상+오디오)
    pyglet 사용, 오디오 딜레이 있지만 에어팟 착용하면 어느정도 보정되는듯
    :param vid_path: 재생할 영상 경로
    :param w: 너비
    :param h: 높이
    :return:
    '''

    # vid_path = 'C:/Users/Dell/Desktop/GSR/시각자극영상/pororo.mp4'
    video = cv2.VideoCapture(vid_path)

    # width, height = 1280, 600  # 영상을 띄울 크기 설정(대충 내 노트북에 맞춤)
    width, height = w, h
    title = "시각자극영상"
    window = pyglet.window.Window(width, height, title)
    player = pyglet.media.Player()
    source = pyglet.media.StreamingSource()
    MediaLoad = pyglet.media.load(vid_path)

    player.queue(MediaLoad)
    player.play()

    # now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 날짜, 시간 출력
    now = datetime.now().strftime("%H-%M-%S.%f")[:-3]  # 시간만 출력
    print("===========시각자극영상===========")
    print("start time: ", now)

    # print("width: ", video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 3
    # print("height: ", video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 4
    print("fps: ", video.get(cv2.CAP_PROP_FPS))  # fps
    print("fps(반올림): ", round(video.get(cv2.CAP_PROP_FPS)))  # fps

    @window.event
    def on_draw():
        if player.source and player.source.video_format:
            player.get_texture().blit(0, 50)  # blit()안의 숫자는 영상을 윈도우 어디서부터 띄울지

    pyglet.app.run()


def record_face(save_path, FPS):
    '''
    RGB, IR카메라로 얼굴 영상 녹화
    녹화 시작 : s/S키
    녹화 종료 : ESC키
    :param save_path: 녹화한 얼굴영상 저장 경로
    :param FPS: 녹화영상 fps 설정
    :return:
    '''

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    record = False

    # 카메라를 추가적으로 연결하여 외장 카메라를 이용하는 경우 장치 번호가 1~n 까지 순차적으로 할당
    ir_capture = cv2.VideoCapture(0)
    rgb_capture = cv2.VideoCapture(1)  # cv2.CAP_DSHOW 쓰면 열리는 속도는 빨라지지만 영상이 왜곡생기고 느려짐
    # laptop_capture = cv2.VideoCapture(0)  # 카메라의 장치 번호(ID)와 연결한다. Index는 카메라의 장치 번호를 의미한다.
    print('카메라 불러옴')

    print("width: ", rgb_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # 프레임 너비
    print("height: ", rgb_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 높이
    print("fps: ", rgb_capture.get(cv2.CAP_PROP_FPS))  # 프레임 속도

    ir_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 1920
    ir_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 1080
    rgb_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    rgb_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # laptop_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # laptop_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print('frame set완료')

    while True:
        # print('---')
        ret, frame = ir_capture.read()
        ret2, frame2 = rgb_capture.read()
        # ret3, frame3 = laptop_capture.read()

        cv2.imshow("ir_vid", frame)
        cv2.imshow('rgb_vid', frame2)
        # cv2.imshow("laptop_vid", frame3)

        # now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # 날짜, 시간 출력
        now = datetime.now().strftime("%H-%M-%S.%f")[:-3]  # 시간만 출력
        key = cv2.waitKey(33)

        # S또는 s를 누르면 시작
        if key == 83 or key == 115:
            print('녹화 시작')
            print('frame', cv2.CAP_PROP_FPS)
            record = True
            # ir_video = cv2.VideoWriter("C:/Users/Dell/Desktop/GSR/gsr/get_videos/" + str(now) + "IR.mp4", fourcc, cv2.CAP_PROP_FPS, (frame.shape[1], frame.shape[0]))
            # rgb_video = cv2.VideoWriter("C:/Users/Dell/Desktop/GSR/gsr/get_videos/" + str(now) + "RGB.mp4", fourcc,cv2.CAP_PROP_FPS, (frame2.shape[1], frame2.shape[0]))

            ir_video = cv2.VideoWriter(save_path + "IR_" + str(now) + ".mp4", fourcc, FPS,
                                       (frame.shape[1], frame.shape[0]))
            rgb_video = cv2.VideoWriter(save_path + "RGB_" + str(now) + ".mp4", fourcc, FPS,
                                        (frame2.shape[1], frame2.shape[0]))
            # laptop_video = cv2.VideoWriter(save_path + "laptop_RGB_" + str(now) + ".mp4", fourcc, FPS, (frame3.shape[1], frame3.shape[0]))

        # ESC 누르면 종료
        elif key == 27:
            print("녹화 중지")
            record = False
            break

        if record == True:
            print("녹화중")
            ir_video.write(frame)
            rgb_video.write(frame2)
            # laptop_video.write(frame3)

    ir_capture.release()
    rgb_capture.release()
    # laptop_capture.release()
    cv2.destroyAllWindows()


