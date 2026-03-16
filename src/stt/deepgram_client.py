"""
Deepgram streaming STT client.

Deepgram is our primary STT provider because of its low-latency
streaming API. The websocket connection stays open for the duration
of the call, and we get interim + final transcription results pushed
back to us as the caller speaks.

Fallback: if Deepgram is down or the API key isn't set, the pipeline
switches to local Whisper automatically (see the factory in main.py).
"""

import asyncio
import time
from typing import AsyncIterator, Optional

from src.stt.base import STTProvider, TranscriptionResult
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

# deepgram SDK import is inside methods so we don't blow up at
# import time if the package isn't installed
try:
    from deepgram import (
        DeepgramClient,
        LiveTranscriptionEvents,
        LiveOptions,
        PrerecordedOptions,
    )
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False
    logger.warning("deepgram SDK not installed — DeepgramSTT will not work")


class DeepgramSTT(STTProvider):
    """
    Real-time speech-to-text via Deepgram's streaming WebSocket API.
    """

    def __init__(self):
        if not DEEPGRAM_AVAILABLE:
            raise RuntimeError(
                "deepgram-sdk is not installed. Run: pip install deepgram-sdk"
            )

        api_key = settings.deepgram_api_key
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is not set in environment")

        self._client = DeepgramClient(api_key)
        self._connection = None
        logger.info("deepgram_stt.initialized")

    async def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        One-shot transcription for batch use. Sends the full audio
        buffer and waits for the result. Not used in the real-time
        path but handy for the evaluation script.
        """
        start = time.monotonic()

        options = PrerecordedOptions(
            model="nova-2",
            language="en",
            smart_format=True,
        )

        payload = {"buffer": audio_bytes, "mimetype": "audio/wav"}
        response = await asyncio.to_thread(
            self._client.listen.prerecorded.v("1").transcribe_file, payload, options
        )

        elapsed = (time.monotonic() - start) * 1000

        # dig into Deepgram's nested response structure
        alt = response.results.channels[0].alternatives[0]
        return TranscriptionResult(
            text=alt.transcript.strip(),
            is_final=True,
            confidence=alt.confidence,
            duration_ms=elapsed,
            language="en",
        )

    async def stream_transcribe(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """
        Open a persistent websocket to Deepgram and pipe audio chunks
        through it. Yields TranscriptionResult objects as they come in.
        """
        result_queue: asyncio.Queue[Optional[TranscriptionResult]] = asyncio.Queue()

        options = LiveOptions(
            model="nova-2",
            language="en",
            encoding="linear16",
            sample_rate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1000,
            vad_events=True,
        )

        connection = self._client.listen.live.v("1")

        # --- event handlers ---
        def on_transcript(_, result, **kwargs):
            """Fires for every interim/final transcript chunk."""
            try:
                alt = result.channel.alternatives[0]
                if not alt.transcript.strip():
                    return
                tr = TranscriptionResult(
                    text=alt.transcript.strip(),
                    is_final=result.is_final,
                    confidence=alt.confidence,
                    duration_ms=result.duration * 1000 if hasattr(result, "duration") else 0,
                    language="en",
                )
                result_queue.put_nowait(tr)
            except (IndexError, AttributeError) as exc:
                logger.warning("deepgram.transcript_parse_error", error=str(exc))

        def on_error(_, error, **kwargs):
            logger.error("deepgram.stream_error", error=str(error))

        def on_close(_, *args, **kwargs):
            # signal the consumer that we're done
            result_queue.put_nowait(None)
            logger.info("deepgram.connection_closed")

        connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        connection.on(LiveTranscriptionEvents.Error, on_error)
        connection.on(LiveTranscriptionEvents.Close, on_close)

        # start the live connection
        await asyncio.to_thread(connection.start, options)
        self._connection = connection
        logger.info("deepgram.stream_started")

        # feed audio in a background task so we can yield results concurrently
        async def _send_audio():
            try:
                async for chunk in audio_stream:
                    connection.send(chunk)
                    # small sleep to avoid hammering the websocket
                    await asyncio.sleep(0.01)
            except Exception as exc:
                logger.error("deepgram.send_error", error=str(exc))
            finally:
                connection.finish()

        send_task = asyncio.create_task(_send_audio())

        # yield results as they arrive
        while True:
            result = await result_queue.get()
            if result is None:
                break
            yield result

        await send_task

    async def close(self) -> None:
        """Shut down any open connection."""
        if self._connection is not None:
            try:
                self._connection.finish()
            except Exception:
                pass
            self._connection = None
        logger.info("deepgram_stt.closed")
