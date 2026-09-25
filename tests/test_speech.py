import io
import wave

import numpy as np

from family_ai.speech import decode_wav, split_for_tts


def test_decode_wav_resamples_to_whisper_rate():
    samples = (np.sin(np.linspace(0, 20, 8000)) * 1000).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(samples.tobytes())
    decoded = decode_wav(output.getvalue())
    assert decoded.dtype == np.float32
    assert 15990 <= len(decoded) <= 16010


def test_tts_splitter_never_exceeds_model_safe_length():
    text = "这是一个需要连续播报的句子。" * 80
    chunks = split_for_tts(text, 150)
    assert len(chunks) > 1
    assert all(len(chunk) <= 150 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == text
