"""Camera boundary: OpenCV BGR is converted to uint8 RGB immediately."""

import cv2


class USBCamera:
    def __init__(self, source: str = "0"):
        self.source = int(source) if source.isdecimal() else source
        self.capture = None

    def __enter__(self):
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera {self.source!r}")
        return self

    def read(self):
        ok, bgr = self.capture.read()
        if not ok or bgr is None:
            raise RuntimeError("Camera failed to return a frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __exit__(self, *_):
        self.capture.release()


def read_rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
