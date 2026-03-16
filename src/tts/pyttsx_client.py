"""
Local TTS fallback using pyttsx3.

This runs entirely on the machine with no API calls, so it's useful
for development and as a fallback if ElevenLabs is down or we're
over quota. The voice quality is basic but functional.

Note: pyttsx3 is synchronous and not thread-safe, so we run it
in a thread pool executor to avoid blocking the event loop.
"""

import asyncio
import io
import tempfile
import time
import os
from typing import Optional

from src.tts.base import TTSProvider, TTSResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logger.warning("pyttsx3 not installed — LocalTTS unavailable")


class LocalTTS(TTSProvider):
    """
    Offline text-to-speech using pyttsx3 (wraps SAPI5/nsss/espeak
    depending on the OS).

    Not great for production calls, but perfectly fine for dev,
    testing, and demos.
    """

    def __init__(self, rate: int = 175):
        if not PYTTSX3_AVAILABLE:
            raise RuntimeError(
                "pyttsx3 is not installed. Run: pip install pyttsx3"
            )

        self._rate = rate
        logger.info("local_tts.initialized", rate=rate)

    def _synthesize_sync(self, text: str) -> tuple:
        """
        Synchronous synthesis — runs in a thread because pyttsx3
        blocks and also doesn't like being called from async context.
        
        Returns (audio_bytes, elapsed_ms).
        """
        engine = pyttsx3.init()
        engine.setProperty("rate", self._rate)

        start = time.monotonic()

        # pyttsx3 can save to file — we use a temp file then read it back
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()

            with open(tmp_path, "rb") as f:
                audio_bytes = f.read()
        finally:
            # clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            engine.stop()

        elapsed = (time.monotonic() - start) * 1000
        return audio_bytes, elapsed

    async def synthesize(self, text: str) -> TTSResult:
        """Generate speech audio from text using local engine."""
        if not text.strip():
            return TTSResult(
                audio_bytes=b"",
                format="wav",
                sample_rate=22050,
                duration_ms=0,
                synthesis_time_ms=0,
            )

        audio_bytes, elapsed_ms = await asyncio.to_thread(
            self._synthesize_sync, text
        )

        # rough duration calc: wav header is 44 bytes, 16-bit mono at 22050 Hz
        pcm_bytes = max(0, len(audio_bytes) - 44)
        duration_ms = (pcm_bytes / (22050 * 2)) * 1000

        logger.info(
            "local_tts.synthesized",
            text_len=len(text),
            audio_bytes=len(audio_bytes),
            latency_ms=round(elapsed_ms, 1),
        )

        return TTSResult(
            audio_bytes=audio_bytes,
            format="wav",
            sample_rate=22050,
            duration_ms=round(duration_ms, 1),
            synthesis_time_ms=round(elapsed_ms, 1),
        )

    async def close(self) -> None:
        """Nothing persistent to clean up."""
        logger.info("local_tts.closed")
