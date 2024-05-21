import threading
import time
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

import cv2
from logger import hmesg_wrapper, iLogger

hmesg = hmesg_wrapper(head="[CAM]")


class CameraStream:
    def __init__(self, device: str = "/dev/video0") -> None:
        self.device = device
        self.lock = RLock()
        self.is_stop = False
        vformat = cv2.CAP_ANY if "rtsp" in device else cv2.CAP_V4L2
        self.cap = cv2.VideoCapture(self.device, vformat)

        self.t: threading.Thread = threading.Thread(
            target=self._read_camera, daemon=True
        )
        self.frame = self.get_first_frame()
        self.t.start()

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        iLogger.info(
            hmesg(
                f"Initialized camera: {self.device}, fps: {self.fps}, resolution: {self.width}x{self.height}"
            )
        )

    def _set_cap(self, resolution, fps: float = 30.0):
        status = self.cap.open(self.path)
        if not status:
            raise RuntimeError(hmesg(f"Can not open the camera from {self.path}"))

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 4.0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        try:
            if resolution:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        except BaseException as e:
            iLogger.debug(hmesg(f"Set camera failed ... {e}"))

        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

    def get_first_frame(self):
        iLogger.debug(hmesg(f"Warnup for camera ({self.device}) ..."))
        ts = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue
            if frame is not None:
                iLogger.debug(
                    hmesg(f"Finished warnup ({self.device}): ({time.time()-ts:.3f}s)")
                )
                return frame
            if time.time - ts > 5:
                raise RuntimeWarning(hmesg(f"Can not open camera ({self.device})"))

    def _update_frame(self, frame):
        with self.lock:
            self.frame = frame

    def _read_camera(self):
        while not self.is_stop:
            ret, frame = self.cap.read()
            if not ret:
                break
            self._update_frame(frame)

        if self.cap.isOpened():
            self.cap.release()

    def _close(self):
        self.is_stop = True
        if self.cap.isOpened():
            self.cap.release()
        iLogger.warning(hmesg(f"Clear camera object ({self.device})"))

    def read(self):
        with self.lock:
            return self.frame

    def release(self):
        self._close()


async def async_init_camera(
    device: str, loop: AbstractEventLoop, executor: ThreadPoolExecutor
):
    return await loop.run_in_executor(executor, CameraStream, device)
