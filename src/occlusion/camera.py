"""Camera boundary: OpenCV BGR is converted to uint8 RGB immediately."""

import cv2
from datetime import datetime, timezone
from time import monotonic_ns


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

    def grab(self):
        """Request a frame before decoding; does not guarantee exposure sync."""
        if not self.capture.grab():
            raise RuntimeError(f"Camera {self.source!r} failed to grab a frame")

    def retrieve_rgb(self):
        ok, bgr = self.capture.retrieve()
        if not ok or bgr is None:
            raise RuntimeError(f"Camera {self.source!r} failed to retrieve a frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __exit__(self, *_):
        self.capture.release()


def read_rgb(path):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def receive_pair(cameras):
    """Grab both before decoding. Host receipt times are not exposure times."""
    frames, timing = {}, {}
    for name in cameras:
        start = monotonic_ns()
        cameras[name].grab()
        timing[name] = {"host_grab_start_ns": start, "host_grab_end_ns": monotonic_ns()}
    for name in cameras:
        frames[name] = cameras[name].retrieve_rgb()
        timing[name].update(host_received_ns=monotonic_ns(),
                            captured_at_utc=datetime.now(timezone.utc).isoformat(),
                            exposure_timestamp=None)
    return frames, timing
