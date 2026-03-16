"""
ElevenLabs TTS client — high-quality neural voice synthesis.

This is our primary TTS path for production because the voice
quality is noticeably better than local engines. Downside is
latency (network round-trip) and cost.

We use their streaming endpoint so we can start playing audio
to the caller before the full response is synthesized.
"""

import asyncio
import time
from typing import Optional

from src.tts.base import TTSProvider, TTSResult
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    logger.warning("elevenlabs SDK not installed — ElevenLabsTTS unavailable")


class ElevenLabsTTS(TTSProvider):
    """
    Text-to-speech via the ElevenLabs API.
    
    We default to their multilingual v2 model because it handles
    conversational English really well, especially with customer
    support phrasing.
    """

    def __init__(
        self,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",
    ):
        if not ELEVENLABS_AVAILABLE:
            raise RuntimeError(
                "elevenlabs SDK not installed. Run: pip install elevenlabs"
            )

        api_key = settings.elevenlabs_api_key
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set in environment")

        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id or settings.elevenlabs_voice_id
        self._model_id = model_id
        logger.info("elevenlabs_tts.initialized", voice_id=self._voice_id)

    async def synthesize(self, text: str) -> TTSResult:
        """
        Send text to ElevenLabs and get back audio bytes.
        
        We collect the full response before returning. For true
        streaming playback see the websocket handler which calls
        generate() directly and streams chunks.
        """
        if not text.strip():
            return TTSResult(
                audio_bytes=b"",
                format="mp3",
                sample_rate=22050,
                duration_ms=0,
                synthesis_time_ms=0,
            )

        start = time.monotonic()

        # the SDK's generate() returns an iterator of audio chunks
        audio_iter = await asyncio.to_thread(
            self._client.generate,
            text=text,
            voice=self._voice_id,
            model=self._model_id,
            output_format="mp3_22050_32",
        )

        # collect all chunks into one buffer
        audio_chunks = []
        for chunk in audio_iter:
            audio_chunks.append(chunk)

        audio_bytes = b"".join(audio_chunks)
        synthesis_ms = (time.monotonic() - start) * 1000

        # rough duration estimate based on MP3 bitrate (32kbps)
        # real duration would need decoding but this is close enough for metrics
        estimated_duration_ms = (len(audio_bytes) * 8 / 32000) * 1000

        logger.info(
            "elevenlabs_tts.synthesized",
            text_len=len(text),
            audio_bytes=len(audio_bytes),
            latency_ms=round(synthesis_ms, 1),
        )

        return TTSResult(
            audio_bytes=audio_bytes,
            format="mp3",
            sample_rate=22050,
            duration_ms=round(estimated_duration_ms, 1),
            synthesis_time_ms=round(synthesis_ms, 1),
        )

    async def close(self) -> None:
        """ElevenLabs client doesn't need explicit cleanup."""
        logger.info("elevenlabs_tts.closed")
