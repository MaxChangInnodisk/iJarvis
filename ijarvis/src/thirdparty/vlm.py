from typing import List

import cv2
import numpy as np
from pydantic.dataclasses import dataclass

from thirdparty import core
from thirdparty.nlp import NLTK

FONT = cv2.FONT_HERSHEY_SIMPLEX
BORDER = cv2.LINE_AA


@dataclass
class GB_Box:
    cx: float
    cy: float
    width: float
    height: float


@dataclass
class GD_Output:
    label: str
    confidence: float
    bbox: GB_Box


class GroundingDinoDrawer:
    def __init__(
        self,
        box_color: list = [0, 255, 0],
        font_color: list = [0, 0, 0],
        alpha: float = 0.3,
        fill_box: int = 0,
    ):
        self.box_color = box_color
        self.font_color = font_color
        self.fill_box = fill_box
        self.alpha = alpha

    def get_bounding_box(self, image: np.ndarray, bbox: GB_Box) -> tuple:
        """return x1, y1, x2, y2"""
        image_height, image_width = image.shape[0], image.shape[1]
        cx = bbox.cx * image_width
        cy = bbox.cy * image_height
        width = bbox.width * image_width
        height = bbox.height * image_height
        x1, y1 = max(0, int(cx - width / 2)), max(0, int(cy - height / 2))
        w, h = int(width), int(height)
        x2, y2 = min(image_width, x1 + w), min(image_height, y1 + h)
        return x1, y1, x2, y2

    def draw(self, image: np.ndarray, results: List[GD_Output]):
        thickness = max(3, image.shape[1] // 1000)
        font_scale = max(0.5, image.shape[1] / 1000)
        font_thickness = max(round(font_scale * 2), image.shape[1] // 1000)

        overlay = image.copy()
        for result in results:
            # Rescale bounding box
            x1, y1, x2, y2 = self.get_bounding_box(image, result.bbox)
            # Define Text
            text = f"{result.label.upper()} ({result.confidence:.2f})"
            (text_width, text_height), base_line = cv2.getTextSize(
                text, FONT, font_scale, font_thickness
            )

            text_x, text_y = (
                x1 + thickness,
                0 + text_height + base_line if y1 - text_height - base_line < 0 else y1,
            )
            # Draw
            if self.fill_box:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.box_color, -1)
                image = cv2.addWeighted(overlay, self.alpha, image, 1 - self.alpha, 0)
                cv2.putText(
                    image,
                    text,
                    (text_x, text_y),
                    FONT,
                    font_scale,
                    self.font_color,
                    font_thickness,
                    BORDER,
                )
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), self.box_color, thickness)
                cv2.rectangle(
                    image,
                    (x1, text_y - text_height - base_line),
                    (x1 + text_width + thickness, text_y + thickness),
                    self.box_color,
                    -1,
                )
                cv2.putText(
                    image,
                    text,
                    (text_x, text_y),
                    FONT,
                    font_scale,
                    self.font_color,
                    font_thickness,
                    BORDER,
                )
        return image


class GroundingDino(core.AsyncApiModel):
    def __init__(self, host: str, port: str, route: str, data_path: str):
        super().__init__(host, port, route)
        self.nltk = NLTK(data_path)

    def _input(
        self, image: str, prompt: str, box_threshold: float, text_threshold: float
    ) -> dict:
        return {
            "image": image,
            "prompt": prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
        }

    def _is_valid_label(self, label: str) -> bool:
        tokens, tagged = self.nltk.inference(content=label)
        if tagged == []:
            return False
        if len(tokens) > 1 and (
            tagged[0][0] == "where"
            or tagged[0][0] == "a"
            or tagged[0][0] == "an"
            or tagged[0][0] == "the"
        ):
            return False
        if (
            tagged[0][1] == "NN"
            or tagged[0][1] == "NNP"
            or tagged[0][1] == "NNS"
            or tagged[0][1] == "NNPS"
        ):
            return True
        else:
            return False

    def _postprocess(self, results: dict) -> List[GD_Output]:
        # Results: boxes, accuracy, object_names
        ret = []
        for idx, box in enumerate(results["boxes"]):
            label = results["object_names"][idx]
            if not self._is_valid_label(label=label):
                continue
            confidence = results["accuracy"][idx]
            gd_box = GB_Box(cx=box[0], cy=box[1], width=box[2], height=box[3])
            ret.append(GD_Output(label=label, confidence=confidence, bbox=gd_box))
        return ret

    async def inference(
        self,
        base64_image: str,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> List[GD_Output]:
        inp = self._input(base64_image, prompt, box_threshold, text_threshold)
        response = await self.client.post(self.url, json=inp)
        response.raise_for_status()  # Raise an HTTPStatusError for bad responses
        response_data = response.json()
        return self._postprocess(response_data["confidence"])
