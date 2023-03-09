import cv2
import time
import threading


class WebcamVideoStream:
    # Opens a video stream with OpenCV from a wired webcam in a thread
    def __init__(self, shape=None, name="WebcamVideoStream"):
        self.name = name
        self.stream = cv2.VideoCapture(0)
        self.shape = shape
        if self.shape is not None:
            self.stream.set(3, shape[0])
            self.stream.set(4, shape[1])
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

    def start(self):
        # Start the thread to read frames from the video stream
        threading.Thread(target=self.update, daemon=True, name=self.name).start()
        return self

    def update(self):
        # Continuosly iterate through the video stream until stopped
        while self.stream.isOpened():
            if not self.stopped():
                self.grabbed, self.frame = self.stream.read()
            else:
                return
        self.stopped
    
    def read(self):
        return self.frame

    def stop(self):
        self.lock.acquire()
        self.stream.release()
        self._stop_event.set()
        self.lock.release()

    def stopped(self):
        return self._stop_event.is_set()