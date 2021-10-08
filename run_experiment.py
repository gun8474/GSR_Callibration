import requests
import pandas as pd
import cv2
import pyglet
from datetime import datetime
from experiment_utils import *


if __name__ == "__main__":
    # 시각자극영상 재생
    vid_path = './시각자극영상/pororo.mp4'
    # vid_path = './시각자극영상/자극영상_타이머/타이머 추가+9분30초.mp4'
    show_video(vid_path, 500, 500)

    # 얼굴 영상 녹화
    save_path = './face_videos/'
    FPS = 30
    record_face(save_path, FPS)

    # neulog 데이터 취득
    neulog_path = './neulog_data'
    file_name = '나연'
    neulog_sensor(neulog_path, file_name)
