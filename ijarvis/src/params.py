import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Union

import thirdparty
import thirdparty.core
import thirdparty.nlp
import thirdparty.vlm
from pydantic import BaseModel
from v4l2 import camera


class WhisperConfig(BaseModel):
    """
    tiny.en -> 3s
    base.en -> 5~6s
    small.en -> 6~9s
    medium.en -> 16~20s
    """

    type: str = "stt"
    name: Literal["tiny.en", "base.en", "small.en", "medium.en"] = "base.en"


class NltkConfig(BaseModel):
    type: str = "nlp"
    path: str = "/usr/local/share/nltk_data"


class GroundingDinoConfig(BaseModel):
    type: str = "vlm"
    host: str = "127.0.0.1"
    port: str = "8000"
    route: str = "/items/"


class MistralConfig(BaseModel):
    type: str = "llm"
    host: str = "127.0.0.1"
    port: str = "5000"
    route: str = "/v1/chat/completions"


class Process(BaseModel):
    status: Literal["init", "doing", "done", "error"] = "init"
    type: Literal["", "stt", "dino", "lama"] = ""
    message: str = "Initialized process object"
    image: str = ""
    performance: dict = {}
    created_time: float = time.time()


@dataclass
class InferenceOptions:
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    box_color: Union[list, tuple] = field(default_factory=lambda: [0, 255, 0])
    text_color: Union[list, tuple] = field(default_factory=lambda: [0, 0, 0])
    fill_box: Optional[int] = 0
    alpha: Optional[float] = 0.3


@dataclass
class Shared:
    exec: Optional[thirdparty.core.AsyncExecutor] = None
    proc: Dict[str, Process] = field(default_factory=dict)
    stt: Optional[thirdparty.stt.Whisper] = None
    llm: Optional[thirdparty.llm.Mistral] = None
    vlm: Optional[thirdparty.vlm.GroundingDino] = None
    nlp: Optional[thirdparty.nlp.AsyncNLTK] = None
    cams: Dict[str, camera.CameraStream] = field(default_factory=dict)
    option: InferenceOptions = InferenceOptions()


shared = Shared()
stt_conf = WhisperConfig()
nlp_conf = NltkConfig()
vlm_conf = GroundingDinoConfig()
llm_conf = MistralConfig()
