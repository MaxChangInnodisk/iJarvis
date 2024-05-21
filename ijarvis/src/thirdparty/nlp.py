import time
from asyncio import AbstractEventLoop
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Coroutine

import nltk
from logger import hmesg_wrapper, iLogger

BGR_COLORS = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "maroon": (0, 0, 128),
    "olive": (0, 128, 128),
    "purple": (128, 0, 128),
    "teal": (128, 128, 0),
    "navy": (128, 0, 0),
}
BGR_COLORS_KEY = list(BGR_COLORS.keys())

hmesg = hmesg_wrapper(head="[NLP]")


class NLTK:
    def __init__(self, data_path: str):
        ts = time.time()
        if not Path(data_path).exists():
            iLogger.info(hmesg("Downloading NLTK Data ..."))

            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")

        self._warmup()

        te = time.time()
        iLogger.info(hmesg(f"NLTK is ready ({te-ts:.3f}s)"))

    def _warmup(self):
        self.inference("Hello World!")

    def inference(self, content: str) -> tuple:
        tokens = nltk.word_tokenize(content)
        tagged = nltk.pos_tag(tokens)
        return (tokens, tagged)


class AsyncNLTK(NLTK):
    def __init__(
        self, data_path: str, loop: AbstractEventLoop, executor: ThreadPoolExecutor
    ):
        super().__init__(data_path)
        self.loop = loop
        self.executor = executor

    async def inference(
        self,
        content: str,
    ) -> Coroutine:
        return await self.loop.run_in_executor(
            self.executor, super().inference, content
        )


async def async_init_async_nltk(
    data_path: str, loop: AbstractEventLoop, executor: ThreadPoolExecutor
):
    return await loop.run_in_executor(executor, AsyncNLTK, data_path, loop, executor)


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_setting_event(nlp_model: NLTK, prompt: str) -> tuple:
    _, tags = nlp_model.inference(prompt)

    keywords = ["color", "threshold"]

    set_keyword = tags[0][0]
    set_item = tags[1][0]

    if set_keyword.lower() not in ["set", "setting"]:
        return None, "not setup event"

    if set_item not in keywords:
        return None, f"expect keywors: {''.join(keywords)}"

    if set_item == "color":
        if tags[-1][1] != "VB" or tags[-1][0] not in BGR_COLORS_KEY:
            return (
                "color",
                f"set color failed, the latest word should be [{BGR_COLORS_KEY}]",
            )
        return "color", BGR_COLORS[tags[-1][0]]

    elif set_item == "threshold":
        if tags[-1][1] != "CD" or not is_float(tags[-1][0]):
            return (
                "threshold",
                "set threshold failed, the latest word should be float (0.0~1.0)",
            )
        return "threshold", tags[-1][0]
