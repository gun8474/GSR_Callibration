import cv2
import numpy as np
from scipy import signal


class RemotePPG:
    """Remote-PPG 추출 및 생리적 파라미터 계산을 위한 클래스

    :param fps: 입력 비디오 프레임율
    :param window_length: 생리적 파라미터 계산을 위한 시간 윈도우의 길이
    :param pass_band: 대역통과 필터링을 위한 주파수 대역(단위: bpm)
    """
    def __init__(self, fps, window_length, pass_band):
        self.fps = fps
        self.pass_band = pass_band
        self.window_length = window_length
        self.buffer = []     # rPPG 처리를 위한 1차원 신호 buffer
        self.live_fps = fps  # 프레임율이 가변적인 라이브 스트림 입력에 대한 실시간 fps
        self.times = []      # live_fps 계산을 위한 프레임별 타임스탬프 저장

    @staticmethod
    def extract_rppg(img, mask):
        """ 얼굴과 얼굴에 대한 피부 마스크를 이용하여 색차 기반 Remote-PPG 추출

        :param img: 얼굴 피부 영상에 대한 2차원 배열
        :param mask: 피부 마스크(img와 동일한 사이즈)
        :return: rPPG 추출 결과값
        """
        # 색공간 변환
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 피부 픽셀 필터링
        if isinstance(mask, type(None)):
            n_pixels = img.shape[0] * img.shape[1]
        else:
            n_pixels = np.sum(mask)
            cr[mask == 0] = 0
            cb[mask == 0] = 0

        # rPPG 값 추출
        value = (np.sum(cr) + np.sum(cb)) / n_pixels
        return value

    @property
    def curr_buffer_size(self):
        # 현재 buffer 길이
        return len(self.buffer)

    def get_buffer(self):
        # 현재 buffer 반환
        return self.buffer

    def update_signal(self, value):
        # buffer 업데이트
        self.buffer.append(value)
        self.buffer = self.buffer[-self.window_length:]

    def update_times(self, time):
        # times 업데이트
        self.times.append(time)
        self.times = self.times[-self.window_length:]

        # 실시간 fps 계산
        if len(self.times) >= 2:
            self.live_fps = len(self.times) / (self.times[-1] - self.times[0])

    @staticmethod
    def detrend_signal(arr, win_size):
        """ 신호 내에 심박수와 관련없는 추세 성분 제거

        :param arr: 대상 시계열 신호
        :param win_size: 추세 제거를 위한 윈도우 크기(일반적으로 비디오 프레임율 사용)
        :return:
        """
        try:
            if not isinstance(win_size, int):
                win_size = int(win_size)
            norm = np.convolve(np.ones(len(arr)), np.ones(win_size), mode='same')
            mean = np.convolve(arr, np.ones(win_size), mode='same') / norm
            return (arr - mean) / mean
        except ValueError:
            return arr

    @staticmethod
    def filter_bandpass(arr, srate, band):
        """ 대역 통과 필터링을 통한 심박 이외의 주파수 성분 제거

        :param arr: 대상 시계열 신호
        :param srate: 샘플링율(=프레임율)
        :param band: 통과 대역(tuple 자료형)
        :return:
        """
        try:
            nyq = 60 * srate / 2
            coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
            return signal.filtfilt(*coef_vector, arr)
        except ValueError:
            return arr

    @staticmethod
    def estimate_heartrate(arr, srate, band):
        """ 입력 시계열 신호로부터 주파수 스펙트럼 분석 기반 심박수 계산

        :param arr: 대상 시계열 신호
        :param srate: 샘플링율(=프레임율)
        :param band: 유효 심박수 범위(단위: bpm)
        :return: 계산된 심박수 값(단위: bpm)
        """
        try:
            pad_factor = max(1, 60 * srate / len(arr))
            n_padded = int(len(arr) * pad_factor)
            f, pxx = signal.periodogram(arr, fs=srate, window='hann', nfft=n_padded)

            max_peak_idx = np.argmax(pxx)
            heartrate = int(f[max_peak_idx] * 60)
            return min(max(heartrate, band[0]), band[1])
        except (ValueError, FloatingPointError):
            return 0

    @staticmethod
    def calculate_ppi(arr, srate, band):
        """ 입력 시계열 신호로부터 peak들을 검출하여 최근 2개 peak 간의 Peak-to-Peak-Interval 계산

        :param arr: 대상 시계열 신호
        :param srate: 샘플링율(프레임율)
        :param band: 유효 심박수 대역(단위: bpm) -> 유효 PPI 범위 계산을 위함
        :return: 계산된 PPI 값(단위: 초)
        """
        ppi = 0.0
        try:
            peaks_idx, _ = signal.find_peaks(arr, height=0.0, distance=int(60 * srate / band[1]))
            if len(peaks_idx) >= 3:
                ppi = (peaks_idx[-2] - peaks_idx[-3]) / srate
                ppi = max(60/band[1], min(60/band[0], ppi))
        except ZeroDivisionError:
            pass

        return ppi
