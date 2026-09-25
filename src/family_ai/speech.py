import io
import re
import tempfile
from threading import Lock
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def split_for_tts(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？.!?；;])\s*", text)
    chunks = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            split_at = max(
                sentence.rfind(mark, 0, max_chars)
                for mark in ("，", ",", "：", ":", " ")
            )
            if split_at < max_chars // 2:
                split_at = max_chars - 1
            part, sentence = sentence[: split_at + 1].strip(), sentence[split_at + 1 :].strip()
            if current:
                chunks.append(current)
                current = ""
            chunks.append(part)
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def decode_wav(data: bytes, max_seconds: int = 60) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as source:
        source.write(data)
        source.flush()
        with wave.open(source.name, "rb") as audio:
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            if sample_width != 2 or channels not in (1, 2) or sample_rate < 8000:
                raise ValueError("仅支持 16-bit PCM WAV 录音")
            if frames / sample_rate > max_seconds:
                raise ValueError(f"单次录音不能超过 {max_seconds} 秒")
            samples = np.frombuffer(audio.readframes(frames), dtype="<i2").astype(np.float32)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    samples /= 32768.0
    if sample_rate != 16000:
        samples = resample_poly(samples, 16000, sample_rate).astype(np.float32)
    return samples


class LocalSpeech:
    def __init__(
        self,
        model_path: Path,
        tts_model_path: Path | None = None,
        chinese_voice: str = "zf_xiaobei",
        english_voice: str = "af_heart",
    ):
        self.model_path = model_path
        self.tts_model_path = tts_model_path
        self.chinese_voice = chinese_voice
        self.english_voice = english_voice
        self._tts_model = None
        self._tts_lock = Lock()

    def warm_transcriber(self) -> None:
        """Load speech-recognition weights before the first household voice request."""
        if not self.model_path.is_dir():
            return
        import mlx_whisper

        mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),
            path_or_hf_repo=str(self.model_path),
            verbose=None,
            language="en",
        )

    def transcribe_wav(self, data: bytes) -> str:
        if not self.model_path.is_dir():
            raise RuntimeError("本地语音识别模型尚未安装")
        import mlx_whisper

        result = mlx_whisper.transcribe(
            decode_wav(data),
            path_or_hf_repo=str(self.model_path),
            verbose=None,
            language=None,
            initial_prompt="Family AI, *, Codex, LangGraph, 家庭助手。",
        )
        return str(result.get("text", "")).strip()

    def synthesize(self, text: str) -> bytes:
        clean = re.sub(r"```.*?```", "", text, flags=re.S)
        clean = re.split(
            r"(?im)^\s*(?:#{1,6}\s*)?(?:消息来源|sources?|references?|来源(?:标题)?|数据来源|参考资料)\s*[:：]?(?:\s+.*)?$",
            clean,
            maxsplit=1,
        )[0]
        clean = re.sub(r"\[([^]]+)]\(https?://[^)]+\)", r"\1", clean)
        clean = re.sub(r"https?://\S+", "", clean)
        clean = re.sub(r"[`*_#>|]", "", clean)
        clean = re.sub(r"(?m)^\s*[-•]\s*", "。", clean)
        clean = re.sub(r"(?m)^\s*(\d+)[.)、]\s*", r"。第\1，", clean)
        clean = clean.replace("°C", "摄氏度").replace("°F", "华氏度").replace("km/h", "公里每小时")
        clean = re.sub(r"\s+", " ", clean).strip()[:1800]
        if not clean:
            raise ValueError("没有可朗读的文字")
        if self.tts_model_path is None or not self.tts_model_path.is_dir():
            raise RuntimeError("本地神经语音模型尚未安装")
        from mlx_audio.tts.utils import load_model

        is_chinese = bool(re.search(r"[\u3400-\u9fff]", clean))
        voice = self.chinese_voice if is_chinese else self.english_voice
        lang_code = "z" if is_chinese else "a"
        with self._tts_lock:
            if self._tts_model is None:
                self._tts_model = load_model(self.tts_model_path)
            pieces = []
            sample_rate = 24000
            chunks = split_for_tts(clean, 180 if is_chinese else 360)
            for index, chunk in enumerate(chunks):
                instruction = (
                    "使用自然、温暖、放松的中文口语，语速适中。像在家里和熟悉的家人轻松聊天，不像新闻主播、客服或机器配音。"
                    if is_chinese
                    else "A warm, natural conversational English voice. Relaxed pace, subtle emotion, no announcer cadence, customer-service tone, or synthetic delivery."
                )
                for result in self._tts_model.generate_voice_design(
                    text=chunk,
                    language="Chinese" if is_chinese else "English",
                    instruct=instruction,
                    temperature=0.7,
                ):
                    sample_rate = result.sample_rate
                    pieces.append(np.asarray(result.audio, dtype=np.float32).reshape(-1))
                if index < len(chunks) - 1:
                    pieces.append(np.zeros(int(sample_rate * 0.09), dtype=np.float32))
        if not pieces:
            raise RuntimeError("神经语音模型没有生成音频")
        audio = np.concatenate(pieces)
        audio = np.clip(audio, -1, 1)
        pcm = (audio * 32767).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm.tobytes())
        return output.getvalue()
