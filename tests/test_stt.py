"""
Tests for the STT module.

These test the interface contract and basic functionality. The actual
API calls are mocked since we don't want to burn API credits in CI.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from src.stt.base import STTProvider, TranscriptionResult


class TestTranscriptionResult:
    """Verify the data structure behaves correctly."""

    def test_creation(self):
        result = TranscriptionResult(
            text="hello world",
            is_final=True,
            confidence=0.95,
            duration_ms=150.0,
        )
        assert result.text == "hello world"
        assert result.is_final is True
        assert result.confidence == 0.95

    def test_optional_language(self):
        result = TranscriptionResult(
            text="test", is_final=True, confidence=0.9, duration_ms=100.0
        )
        assert result.language is None

        result_with_lang = TranscriptionResult(
            text="test", is_final=True, confidence=0.9,
            duration_ms=100.0, language="en"
        )
        assert result_with_lang.language == "en"


class TestSTTProviderInterface:
    """Make sure the abstract base works as expected."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            STTProvider()

    def test_concrete_implementation_required(self):
        # a minimal implementation should work
        class MinimalSTT(STTProvider):
            async def transcribe(self, audio_bytes):
                return TranscriptionResult(
                    text="test", is_final=True, confidence=1.0, duration_ms=0
                )

            async def stream_transcribe(self, audio_stream):
                yield TranscriptionResult(
                    text="test", is_final=True, confidence=1.0, duration_ms=0
                )

            async def close(self):
                pass

        stt = MinimalSTT()
        assert stt is not None


class TestWhisperSTT:
    """Test Whisper STT with mocked model."""

    @patch("src.stt.whisper_client.WHISPER_AVAILABLE", True)
    @patch("src.stt.whisper_client.whisper")
    def test_init(self, mock_whisper):
        mock_whisper.load_model.return_value = MagicMock()

        from src.stt.whisper_client import WhisperSTT
        stt = WhisperSTT(model_name="tiny")
        assert stt._model_name == "tiny"

    @pytest.mark.asyncio
    @patch("src.stt.whisper_client.WHISPER_AVAILABLE", True)
    @patch("src.stt.whisper_client.whisper")
    async def test_transcribe(self, mock_whisper):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "text": "I need help with my account",
            "segments": [{"avg_logprob": -0.3}],
        }
        mock_whisper.load_model.return_value = mock_model

        from src.stt.whisper_client import WhisperSTT
        stt = WhisperSTT(model_name="tiny")

        # 1 second of silence as test audio
        audio = b"\x00" * 32000
        result = await stt.transcribe(audio)

        assert result.text == "I need help with my account"
        assert result.is_final is True
        assert result.confidence > 0
