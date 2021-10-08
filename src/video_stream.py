import cv2


class VideoStream:
    """ 비디오 스트림 관리 클래스 (비디오 파일 또는 라이브 카메라 연결 지원)

    :param src: 카메라 디바이스 번호 또는 비디오 파일 경로
    """
    def __init__(self, src):
        # 비디오 스트림 생성
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise IOError

        # 입력 데이터에 따른 초기화 수행
        if isinstance(src, str):          # 입력이 비디오 파일인 경우
            self.is_file = True
            self.n_frames = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))  # 총 프레임 개수
        else:
            if not isinstance(src, int):  # 입력이 라이브 스트림인 경우
                raise TypeError
            self.is_file = False
            self.n_frames = -1

        # 비디오의 height, width 정보
        self.width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        # frame 단위로 읽어오기
        return self.stream.read()

    def close(self):
        # 리소스 정리
        self.stream.release()


