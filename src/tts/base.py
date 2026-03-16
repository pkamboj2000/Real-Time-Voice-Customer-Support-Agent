"""
Abstract base for text-to-speech providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TTSResult:
    """Output from a TTS synthesis call."""
    audio_bytes: bytes          # raw audio (PCM or MP3 depending on provider)
    format: str                 # "pcm", "mp3", "wav"
    sample_rate: int
    duration_ms: float          # how long the generated audio is
    synthesis_time_ms: float    # how long the API / local engine took


class TTSProvider(ABC):
    """
    All TTS providers share this interface so we can swap them
    without changing the rest of the pipeline.
    """

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """Convert text to speech audio."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
        ...
