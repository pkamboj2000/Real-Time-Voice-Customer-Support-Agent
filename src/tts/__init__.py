# tts package
from src.tts.base import TTSProvider
from src.tts.elevenlabs_client import ElevenLabsTTS
from src.tts.pyttsx_client import LocalTTS

__all__ = ["TTSProvider", "ElevenLabsTTS", "LocalTTS"]
