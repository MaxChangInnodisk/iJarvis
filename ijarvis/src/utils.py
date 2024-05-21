import base64
import copy
import threading
import time

import cv2
import numpy as np
from logger import hmesg_wrapper, iLogger
from params import shared

hmesg = hmesg_wrapper(head="[UTIL]")


def str_to_cv(base64_image: str):
    image_data = base64.b64decode(base64_image)
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def cv_to_str(image: np.ndarray):
    _, image_buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(image_buffer).decode("utf-8")


def buffer_to_cv(buffer: bytes):
    nparr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def cv_to_buffer(image: np.ndarray) -> bytes:
    _, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return buffer.tobytes()


def head_message(text: str, head: str = "[ijarvis]") -> str:
    return f"{head} {text}"


class SharedKiller:
    def __init__(self, interval: int = 5, lifetime: int = 10):
        self.is_stop = False
        self.interval = interval
        self.lifetime = lifetime

        self.t = threading.Thread(target=self.event, daemon=True)
        self.t.start()
        iLogger.info(hmesg(f"Init Killer, process life time: {lifetime}s"))

    def event(self):
        while not self.is_stop:
            time.sleep(self.interval)
            tcur = time.time()
            aliave_proc_nums = 0
            kill_proc_nums = 0
            processes = copy.copy(shared.proc)
            for uuid, proc in processes.items():
                t_proc_exist = tcur - proc.created_time
                if t_proc_exist < self.lifetime:
                    aliave_proc_nums += 1
                    continue
                shared.proc.pop(uuid)
                kill_proc_nums += 1
            # print(
            #     hmesg(f"Process killer, A/K: {aliave_proc_nums}/{kill_proc_nums}"),
            #     end="\r",
            # )
        iLogger.info(hmesg("Stop killer"))

    def release(self):
        self.is_stop = True
