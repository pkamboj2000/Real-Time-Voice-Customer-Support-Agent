"""
Local Whisper-based STT — our fallback when Deepgram is unavailable.

Uses OpenAI's open-source Whisper model running locally. Slower than
Deepgram's streaming API but works completely offline and doesn't
require an API key.

For real-time use we process audio in small batches (not true streaming)
because Whisper works on fixed-length windows. We accumulate audio
until we have enough for a reasonable transcription chunk (~2-3 seconds),
then run inference on it.
"""

import asyncio
import io
import tempfile
import time
from typing import AsyncIterator, Optional

import numpy as np

from src.stt.base import STTProvider, TranscriptionResult
from src.utils.audio import pcm_to_float32, create_wav_header
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("whisper not installed — WhisperSTT unavailable")


class WhisperSTT(STTProvider):
    """
    Local speech-to-text using OpenAI Whisper.

    The model is loaded once and kept in memory for the life of the
    process. We default to 'base' which balances speed and accuracy
    well enough for English customer support calls.
    """

    def __init__(self, model_name: str = "base"):
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "openai-whisper is not installed. Run: pip install openai-whisper"
            )

        logger.info("whisper_stt.loading_model", model=model_name)
        self._model = whisper.load_model(model_name)
        self._model_name = model_name

        # accumulation buffer for simulated streaming
        # whisper needs at least ~1s of audio to produce anything useful
        self._min_chunk_seconds = 2.0
        self._sample_rate = settings.audio_sample_rate

        logger.info("whisper_stt.model_loaded", model=model_name)

    async def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        Transcribe a full audio buffer. We write it to a temp file
        because whisper.transcribe() expects a file path.
        """
        start = time.monotonic()

        # wrap raw PCM in a WAV header so whisper can read it
        wav_data = create_wav_header(
            sample_rate=self._sample_rate,
            data_size=len(audio_bytes)
        ) + audio_bytes

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(wav_data)
            tmp.flush()

            result = await asyncio.to_thread(
                self._model.transcribe,
                tmp.name,
                language="en",
                fp16=False,
            )

        elapsed = (time.monotonic() - start) * 1000
        text = result.get("text", "").strip()

        # whisper doesn't give us per-segment confidence directly,
        # but we can approximate from the avg logprob across segments
        segments = result.get("segments", [])
        avg_confidence = 0.0
        if segments:
            logprobs = [seg.get("avg_logprob", -1.0) for seg in segments]
            # convert log prob to a rough 0-1 confidence
            import math
            avg_confidence = min(1.0, max(0.0, math.exp(sum(logprobs) / len(logprobs))))

        return TranscriptionResult(
            text=text,
            is_final=True,
            confidence=round(avg_confidence, 3),
            duration_ms=round(elapsed, 2),
            language="en",
        )

    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Simulated streaming: accumulate audio chunks until we have
        enough for a meaningful transcription, then run Whisper on
        that batch and yield the result.

        This is obviously higher-latency than Deepgram streaming, but
        it works offline and is useful as a fallback.
        """
        buffer = bytearray()
        bytes_per_second = self._sample_rate * 2  # 16-bit = 2 bytes/sample
        min_bytes = int(self._min_chunk_seconds * bytes_per_second)

        async for chunk in audio_stream:
            buffer.extend(chunk)

            if len(buffer) >= min_bytes:
                audio_snapshot = bytes(buffer)
                buffer.clear()

                result = await self.transcribe(audio_snapshot)
                if result.text:
                    yield result

        # whatever is left in the buffer at the end
        if len(buffer) > 0:
            result = await self.transcribe(bytes(buffer))
            if result.text:
                yield result

    async def close(self) -> None:
        """Nothing to clean up for local whisper — model lives in memory."""
        logger.info("whisper_stt.closed")
