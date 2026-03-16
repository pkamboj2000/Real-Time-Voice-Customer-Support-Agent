"""
Audio utility helpers — format conversion, chunking, and VAD (voice activity detection).

Most of the real-time pipeline works with raw 16-bit PCM at 16kHz mono,
but callers might send different formats through the websocket. These
helpers smooth over the differences.
"""

import io
import struct
from typing import Iterator, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def pcm_to_float32(pcm_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Convert raw PCM bytes to a float32 numpy array normalized to [-1.0, 1.0].
    Assumes little-endian byte order (standard for WAV / most telephony).
    """
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    samples = np.frombuffer(pcm_bytes, dtype=dtype)
    return samples.astype(np.float32) / np.iinfo(dtype).max


def float32_to_pcm(audio: np.ndarray, sample_width: int = 2) -> bytes:
    """Inverse of pcm_to_float32 — back to raw PCM bytes."""
    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    scaled = (audio * np.iinfo(dtype).max).clip(
        np.iinfo(dtype).min, np.iinfo(dtype).max
    )
    return scaled.astype(dtype).tobytes()


def chunk_audio(
    audio_bytes: bytes,
    chunk_size_ms: int = 100,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
) -> Iterator[bytes]:
    """
    Split a continuous audio buffer into fixed-duration chunks.
    Useful when we receive a big blob and need to stream it downstream.
    """
    bytes_per_sample = sample_width * channels
    bytes_per_chunk = int(sample_rate * (chunk_size_ms / 1000.0)) * bytes_per_sample

    offset = 0
    while offset + bytes_per_chunk <= len(audio_bytes):
        yield audio_bytes[offset : offset + bytes_per_chunk]
        offset += bytes_per_chunk

    # send remaining tail if there's anything left
    if offset < len(audio_bytes):
        yield audio_bytes[offset:]


def compute_rms(pcm_bytes: bytes, sample_width: int = 2) -> float:
    """
    Quick RMS energy calculation — handy for silence detection
    before we bother sending audio to the STT provider.
    """
    samples = pcm_to_float32(pcm_bytes, sample_width)
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples ** 2)))


def is_speech(
    pcm_bytes: bytes,
    threshold: float = 0.01,
    sample_width: int = 2,
) -> bool:
    """
    Simple energy-based speech/silence gate. Not as good as webrtcvad
    but works as a quick pre-filter to avoid sending dead air to the
    STT service (which can eat up quota).
    """
    rms = compute_rms(pcm_bytes, sample_width)
    return rms > threshold


def resample_if_needed(
    audio: np.ndarray,
    current_rate: int,
    target_rate: int = 16000,
) -> np.ndarray:
    """
    Resample audio to the target sample rate. Uses linear interpolation
    which is far from perfect but good enough for speech at telephony
    quality. For anything more serious we'd pull in librosa.resample.
    """
    if current_rate == target_rate:
        return audio

    ratio = target_rate / current_rate
    target_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, target_length)
    resampled = np.interp(indices, np.arange(len(audio)), audio)
    return resampled.astype(np.float32)


def create_wav_header(
    sample_rate: int = 16000,
    bits_per_sample: int = 16,
    channels: int = 1,
    data_size: int = 0,
) -> bytes:
    """
    Build a minimal WAV header. We use this when piping raw PCM
    into libraries that insist on a proper WAV file.
    """
    byte_rate = sample_rate * channels * (bits_per_sample // 8)
    block_align = channels * (bits_per_sample // 8)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM subchunk size
        1,   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header
