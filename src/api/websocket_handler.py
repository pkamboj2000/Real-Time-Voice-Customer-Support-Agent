"""
WebSocket handler for real-time voice communication.

This is the hot path: audio comes in from the caller, gets transcribed,
processed by the agent, converted to speech, and sent back — all over
a single persistent WebSocket connection.

Protocol:
  Client → Server:
    - Binary frames: raw audio (16-bit PCM, 16kHz, mono)
    - Text frames: JSON control messages (e.g., {"type": "end_call"})

  Server → Client:
    - Binary frames: audio response (MP3 or WAV depending on TTS provider)
    - Text frames: JSON status updates (transcript, intent, etc.)
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.agent.graph import VoiceAgent
from src.stt.base import STTProvider
from src.tts.base import TTSProvider
from src.analytics.metrics import MetricsCollector
from src.utils.audio import is_speech
from src.utils.transcript import TranscriptStore
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

ws_router = APIRouter()

# injected from main.py
_agent: Optional[VoiceAgent] = None
_stt: Optional[STTProvider] = None
_tts: Optional[TTSProvider] = None
_metrics: Optional[MetricsCollector] = None


def set_ws_dependencies(
    agent: VoiceAgent,
    stt: STTProvider,
    tts: TTSProvider,
    metrics: MetricsCollector,
):
    """Called from main.py to inject the shared service instances."""
    global _agent, _stt, _tts, _metrics
    _agent = agent
    _stt = stt
    _tts = tts
    _metrics = metrics


class CallSession:
    """
    Manages the state of a single active call over WebSocket.

    Handles the full lifecycle: accumulating audio, feeding it to STT,
    running the agent pipeline, calling TTS, and sending audio back.
    Also manages barge-in detection (caller interrupting the agent).
    """

    def __init__(self, websocket: WebSocket, session_id: str):
        self.websocket = websocket
        self.session_id = session_id
        self.transcript = TranscriptStore()
        self.conversation_history: List[Dict[str, str]] = []
        self.is_agent_speaking = False
        self.barge_in_detected = False
        self._audio_buffer = bytearray()
        self._silence_frames = 0
        # how many consecutive silence frames before we consider the user done talking
        self._silence_threshold = 15  # about 1.5 seconds at 100ms chunks

    async def send_status(self, event_type: str, data: Dict = None):
        """Send a JSON status message to the client."""
        msg = {"type": event_type, "session_id": self.session_id}
        if data:
            msg.update(data)
        try:
            await self.websocket.send_text(json.dumps(msg))
        except Exception as exc:
            logger.warning("ws.send_failed", error=str(exc))

    def add_audio(self, audio_chunk: bytes) -> bool:
        """
        Buffer incoming audio. Returns True when we have enough
        to process (caller stopped talking).
        """
        if is_speech(audio_chunk):
            self._audio_buffer.extend(audio_chunk)
            self._silence_frames = 0

            # barge-in: if the agent is currently speaking and
            # the caller starts talking, we should stop playback
            if self.is_agent_speaking:
                self.barge_in_detected = True
                logger.info("ws.barge_in_detected", session=self.session_id)

            return False
        else:
            self._silence_frames += 1
            # only trigger processing if we have meaningful audio
            if self._silence_frames >= self._silence_threshold and len(self._audio_buffer) > 0:
                return True
            return False

    def get_and_clear_audio(self) -> bytes:
        """Retrieve the accumulated audio buffer and reset it."""
        audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._silence_frames = 0
        return audio

    def add_to_history(self, role: str, content: str):
        """Append a turn to the conversation history."""
        self.conversation_history.append({
            "role": "user" if role == "caller" else "assistant",
            "content": content,
        })


@ws_router.websocket("/ws/call")
async def handle_call(websocket: WebSocket):
    """
    Main WebSocket endpoint for voice calls.

    The flow:
    1. Client connects and starts streaming audio
    2. We buffer audio until the caller pauses
    3. Run STT on the buffered audio
    4. Feed the transcript into the agent pipeline
    5. Run TTS on the agent's response
    6. Stream the audio back
    7. Repeat until the call ends
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session = CallSession(websocket, session_id)

    logger.info("ws.call_started", session=session_id)
    await session.send_status("call_started", {"message": "Connected to voice agent"})

    try:
        while True:
            data = await websocket.receive()

            # handle text control messages
            if "text" in data:
                try:
                    control = json.loads(data["text"])
                    if control.get("type") == "end_call":
                        logger.info("ws.call_ended_by_client", session=session_id)
                        break
                except json.JSONDecodeError:
                    pass
                continue

            # handle binary audio frames
            if "bytes" in data:
                audio_chunk = data["bytes"]
                caller_done = session.add_audio(audio_chunk)

                if caller_done:
                    await _process_turn(session)

    except WebSocketDisconnect:
        logger.info("ws.disconnected", session=session_id)
    except Exception as exc:
        logger.error("ws.error", session=session_id, error=str(exc))
    finally:
        # save transcript regardless of how the call ended
        transcript_data = session.transcript.finalize(
            resolved=not session.transcript.session.escalated
        )
        session.transcript.save_to_disk()

        if _metrics is not None:
            _metrics.record_session_complete(
                session_id=session_id,
                turn_count=session.transcript.session.turn_count,
                avg_latency=session.transcript.session.avg_latency_ms,
                resolved=session.transcript.session.resolved,
                escalated=session.transcript.session.escalated,
            )

        await session.send_status("call_ended", {
            "turns": session.transcript.session.turn_count,
            "resolved": session.transcript.session.resolved,
        })

        logger.info(
            "ws.call_complete",
            session=session_id,
            turns=session.transcript.session.turn_count,
        )


async def _process_turn(session: CallSession):
    """
    Process one caller turn: STT → Agent → TTS → send back.
    """
    if _stt is None or _agent is None or _tts is None:
        await session.send_status("error", {"message": "Services not ready"})
        return

    turn_start = time.monotonic()

    # Step 1: speech to text
    audio_bytes = session.get_and_clear_audio()
    if not audio_bytes:
        return

    await session.send_status("processing", {"step": "transcribing"})

    try:
        stt_result = await _stt.transcribe(audio_bytes)
    except Exception as exc:
        logger.error("ws.stt_failed", error=str(exc), session=session.session_id)
        await session.send_status("error", {"message": "Transcription failed"})
        return

    caller_text = stt_result.text
    if not caller_text.strip():
        return

    session.transcript.add_caller_turn(caller_text)
    session.add_to_history("caller", caller_text)

    await session.send_status("transcript", {
        "role": "caller",
        "text": caller_text,
        "confidence": stt_result.confidence,
    })

    # Step 2: agent processing
    await session.send_status("processing", {"step": "thinking"})

    result = _agent.process_message(
        user_message=caller_text,
        session_id=session.session_id,
        conversation_history=session.conversation_history,
    )

    agent_text = result.get("agent_response", "")
    intent = result.get("intent", "")
    confidence = result.get("confidence", 0.0)
    escalated = result.get("should_escalate", False)

    # record in transcript
    agent_latency = result.get("total_latency_ms", 0.0)
    sources = [d.get("source", "") for d in result.get("retrieved_docs", [])]

    session.transcript.add_agent_turn(
        text=agent_text,
        intent=intent,
        confidence=confidence,
        latency_ms=agent_latency,
        escalated=escalated,
        retrieval_sources=sources,
    )
    session.add_to_history("agent", agent_text)

    await session.send_status("transcript", {
        "role": "agent",
        "text": agent_text,
        "intent": intent,
        "confidence": confidence,
        "escalated": escalated,
    })

    # Step 3: text to speech
    await session.send_status("processing", {"step": "speaking"})
    session.is_agent_speaking = True
    session.barge_in_detected = False

    try:
        tts_result = await _tts.synthesize(agent_text)

        # only send audio if the caller hasn't interrupted
        if not session.barge_in_detected and tts_result.audio_bytes:
            await session.websocket.send_bytes(tts_result.audio_bytes)
    except Exception as exc:
        logger.error("ws.tts_failed", error=str(exc), session=session.session_id)

    session.is_agent_speaking = False

    # total turn timing
    total_ms = (time.monotonic() - turn_start) * 1000

    await session.send_status("turn_complete", {
        "latency_ms": round(total_ms, 2),
        "step_latencies": result.get("step_latencies", {}),
    })

    if _metrics is not None:
        _metrics.record_request(
            session_id=session.session_id,
            intent=intent,
            latency_ms=total_ms,
            escalated=escalated,
            resolved=not escalated,
        )
