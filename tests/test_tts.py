"""
Tests for the TTS module.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.tts.base import TTSProvider, TTSResult


class TestTTSResult:
    def test_creation(self):
        result = TTSResult(
            audio_bytes=b"fake audio",
            format="mp3",
            sample_rate=22050,
            duration_ms=500.0,
            synthesis_time_ms=120.0,
        )
        assert result.format == "mp3"
        assert result.sample_rate == 22050
        assert len(result.audio_bytes) == 10


class TestTTSProviderInterface:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            TTSProvider()

    def test_concrete_implementation(self):
        class DummyTTS(TTSProvider):
            async def synthesize(self, text):
                return TTSResult(
                    audio_bytes=b"audio",
                    format="wav",
                    sample_rate=16000,
                    duration_ms=100,
                    synthesis_time_ms=50,
                )
            async def close(self):
                pass

        tts = DummyTTS()
        assert tts is not None


class TestLocalTTS:
    """Test the pyttsx3-based local TTS."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self):
        """Synthesizing empty text should return empty audio."""
        from src.tts.pyttsx_client import LocalTTS, PYTTSX3_AVAILABLE

        if not PYTTSX3_AVAILABLE:
            pytest.skip("pyttsx3 not installed")

        tts = LocalTTS()
        result = await tts.synthesize("")
        assert result.audio_bytes == b""
        assert result.duration_ms == 0

    @pytest.mark.asyncio
    async def test_synthesize_produces_audio(self):
        """Basic check that synthesis produces non-empty bytes."""
        from src.tts.pyttsx_client import LocalTTS, PYTTSX3_AVAILABLE

        if not PYTTSX3_AVAILABLE:
            pytest.skip("pyttsx3 not installed")

        tts = LocalTTS()
        result = await tts.synthesize("Hello, how can I help you today?")
        assert len(result.audio_bytes) > 0
        assert result.format == "wav"
        assert result.synthesis_time_ms > 0
