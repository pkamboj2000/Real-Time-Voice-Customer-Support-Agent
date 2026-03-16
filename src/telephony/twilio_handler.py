"""
Twilio telephony integration — connects real phone calls to our WebSocket pipeline.

This module handles:
  1. Incoming call webhook (Twilio hits our /twilio/voice endpoint)
  2. TwiML response that redirects the call to our media stream WebSocket
  3. Twilio Media Streams WebSocket handler (receives audio, sends audio back)

The Twilio Media Streams protocol is different from our internal WebSocket
protocol — it wraps audio in JSON messages with base64 encoding. This
handler translates between the two formats.

Setup (in Twilio console):
  1. Buy a phone number
  2. Set the voice webhook to https://your-domain.com/twilio/voice
  3. Make sure your server is publicly accessible (ngrok for local dev)
"""

import base64
import json
import asyncio
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response

from src.agent.graph import VoiceAgent
from src.stt.base import STTProvider
from src.tts.base import TTSProvider
from src.analytics.metrics import MetricsCollector
from src.utils.transcript import TranscriptStore
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)

twilio_router = APIRouter(prefix="/twilio")

# injected from main.py
_agent: Optional[VoiceAgent] = None
_stt: Optional[STTProvider] = None
_tts: Optional[TTSProvider] = None
_metrics: Optional[MetricsCollector] = None


def set_twilio_dependencies(
    agent: VoiceAgent,
    stt: STTProvider,
    tts: TTSProvider,
    metrics: MetricsCollector,
):
    global _agent, _stt, _tts, _metrics
    _agent = agent
    _stt = stt
    _tts = tts
    _metrics = metrics


@twilio_router.post("/voice")
async def handle_incoming_call(request: Request):
    """
    Twilio webhook for incoming voice calls.

    Responds with TwiML that opens a bidirectional media stream
    back to our WebSocket endpoint. Twilio handles the SIP/PSTN
    side; we just deal with audio over the websocket.
    """
    # figure out our public URL from the request headers
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.url.scheme == "https" else "ws"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>Thank you for calling TechFlow Solutions. One moment while I connect you with our support assistant.</Say>
    <Connect>
        <Stream url="{scheme}://{host}/twilio/media-stream">
            <Parameter name="caller_number" value="{{{{From}}}}"/>
        </Stream>
    </Connect>
</Response>"""

    logger.info("twilio.incoming_call")
    return Response(content=twiml, media_type="application/xml")


@twilio_router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """
    Twilio Media Streams WebSocket endpoint.

    Twilio sends audio as base64-encoded mulaw at 8kHz in JSON
    messages. We decode it, process it through our pipeline, and
    send audio back the same way.

    Protocol reference: https://www.twilio.com/docs/voice/media-streams
    """
    await websocket.accept()
    stream_sid = None
    caller_number = None
    transcript = TranscriptStore()
    audio_buffer = bytearray()
    conversation_history = []

    logger.info("twilio.media_stream_connected")

    try:
        async for message in websocket.iter_text():
            data = json.loads(message)
            event = data.get("event")

            if event == "connected":
                logger.info("twilio.stream_connected")

            elif event == "start":
                # stream metadata
                start_data = data.get("start", {})
                stream_sid = start_data.get("streamSid")
                custom_params = start_data.get("customParameters", {})
                caller_number = custom_params.get("caller_number", "unknown")
                transcript = TranscriptStore(caller_number=caller_number)
                logger.info(
                    "twilio.stream_started",
                    stream_sid=stream_sid,
                    caller=caller_number,
                )

            elif event == "media":
                # audio data — base64-encoded mulaw
                payload = data.get("media", {})
                audio_b64 = payload.get("payload", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    audio_buffer.extend(audio_bytes)

                    # process when we have ~2 seconds of audio (mulaw at 8kHz = 8000 bytes/sec)
                    if len(audio_buffer) >= 16000:
                        await _process_twilio_audio(
                            websocket=websocket,
                            stream_sid=stream_sid,
                            audio_bytes=bytes(audio_buffer),
                            transcript=transcript,
                            conversation_history=conversation_history,
                        )
                        audio_buffer.clear()

            elif event == "stop":
                logger.info("twilio.stream_stopped", stream_sid=stream_sid)
                break

    except WebSocketDisconnect:
        logger.info("twilio.stream_disconnected", stream_sid=stream_sid)
    except Exception as exc:
        logger.error("twilio.stream_error", error=str(exc))
    finally:
        transcript.finalize(resolved=not transcript.session.escalated)
        transcript.save_to_disk()


async def _process_twilio_audio(
    websocket: WebSocket,
    stream_sid: Optional[str],
    audio_bytes: bytes,
    transcript: TranscriptStore,
    conversation_history: list,
):
    """
    Process a chunk of audio from Twilio's media stream.

    The audio comes in as mulaw 8kHz which most STT providers
    can handle directly. Response audio needs to be sent back
    as mulaw too.
    """
    if _stt is None or _agent is None or _tts is None:
        return

    # transcribe
    try:
        stt_result = await _stt.transcribe(audio_bytes)
    except Exception as exc:
        logger.error("twilio.stt_failed", error=str(exc))
        return

    caller_text = stt_result.text
    if not caller_text.strip():
        return

    transcript.add_caller_turn(caller_text)
    conversation_history.append({"role": "user", "content": caller_text})

    # agent pipeline
    result = _agent.process_message(
        user_message=caller_text,
        session_id=transcript.session_id,
        conversation_history=conversation_history,
    )

    agent_text = result.get("agent_response", "")
    transcript.add_agent_turn(
        text=agent_text,
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        latency_ms=result.get("total_latency_ms", 0.0),
        escalated=result.get("should_escalate", False),
    )
    conversation_history.append({"role": "assistant", "content": agent_text})

    # synthesize and send back
    try:
        tts_result = await _tts.synthesize(agent_text)

        if tts_result.audio_bytes and stream_sid:
            # encode audio as base64 and wrap in Twilio's media message format
            audio_b64 = base64.b64encode(tts_result.audio_bytes).decode("utf-8")
            media_message = json.dumps({
                "event": "media",
                "streamSid": stream_sid,
                "media": {
                    "payload": audio_b64,
                },
            })
            await websocket.send_text(media_message)

    except Exception as exc:
        logger.error("twilio.tts_send_failed", error=str(exc))
