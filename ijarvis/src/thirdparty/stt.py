import time
import wave
from pathlib import Path
from typing import Coroutine, Union

import numpy as np
import torch
import whisper
from logger import hmesg_wrapper, iLogger

from thirdparty.core import AsyncExecutor, InferenceModel

TMP_WAV = Path("/tmp/warmup.wav")
hmesg = hmesg_wrapper(head="[STT]")


class Whisper(InferenceModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        iLogger.info(hmesg(f"Loading model ({self.model_name}) with {self.device}"))

        ts = time.time()
        self.model = whisper.load_model(self.model_name, device=self.device)

        warmup_audio_path = mock_wav()
        if warmup_audio_path is not None and warmup_audio_path.exists():
            self.model.transcribe(str(warmup_audio_path))["text"]

        iLogger.info(hmesg(f"The Whisper model is ready {time.time()-ts:.3f}s"))

    def inference(self, audio_path: Path) -> str:
        try:
            return self.model.transcribe(str(audio_path))["text"]
        except BaseException as e:
            iLogger.error(hmesg(str(e)))
            return ""

    def release(self):
        del self.model


class AsyncWhisper(Whisper):
    def __init__(
        self,
        model_name: str,
        executor: AsyncExecutor,
    ):
        Whisper.__init__(self, model_name)
        self.exec = executor

    async def inference(
        self,
        audio_path: Path,
    ) -> Coroutine:
        return await self.exec(super().inference, str(audio_path))


def mock_wav() -> Union[None, Path]:
    # 設置參數
    sample_rate = 44100  # 采樣率
    duration = 1.0  # 持續時間，以秒為單位
    frequency = 440.0  # 正弦波頻率，這裡是 A4 音高
    # 生成音頻數據
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
    # 轉換為 16 位 PCM 格式
    audio_data = (audio_data * 32767).astype(np.int16)
    # 創建 WAV 文件
    with wave.open(str(TMP_WAV), "w") as wav_file:
        # 設置 WAV 文件參數
        wav_file.setnchannels(1)  # 單聲道
        wav_file.setsampwidth(2)  # 每樣本寬度，2 字節，即 16 位
        wav_file.setframerate(sample_rate)  # 設置采樣率
        # 寫入音頻數據
        wav_file.writeframes(audio_data.tobytes())

    if TMP_WAV.exists():
        return TMP_WAV

    return None
