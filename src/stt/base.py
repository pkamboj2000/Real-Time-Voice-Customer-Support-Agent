"""
Abstract base for speech-to-text providers.

The idea is that we can swap Deepgram for Whisper (or anything else)
without touching the rest of the pipeline. Each provider implements
the same interface, and the orchestrator just calls transcribe().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class TranscriptionResult:
    """Wrapper around a single STT result chunk."""
    text: str
    is_final: bool          # whether this is a final or interim result
    confidence: float       # 0.0 to 1.0
    duration_ms: float      # how long the audio segment was
    language: Optional[str] = None


class STTProvider(ABC):
    """
    Base class all STT providers must implement.

    There are two modes:
    1. transcribe() — process a complete audio buffer at once (batch)
    2. stream_transcribe() — process audio chunks as they arrive (real-time)
    """

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        Transcribe a complete audio buffer and return the result.
        Used mainly for testing and the evaluation suite.
        """
        ...

    @abstractmethod
    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Accept a stream of audio chunks and yield transcription results
        as they become available. This is the hot path in production.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up any persistent connections or resources."""
        ...
