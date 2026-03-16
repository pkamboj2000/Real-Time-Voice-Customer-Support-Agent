"""
Application entry point — ties everything together.

Creates the FastAPI app, initializes all services (STT, TTS, agent,
vector store), and wires up the routes. Run with:

    uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_dependencies
from src.api.websocket_handler import ws_router, set_ws_dependencies
from src.api.middleware import RequestLoggingMiddleware
from src.telephony.twilio_handler import twilio_router, set_twilio_dependencies
from src.agent.graph import VoiceAgent
from src.analytics.metrics import MetricsCollector
from src.analytics.dashboard import DashboardService
from src.retrieval.vector_store import get_vector_store
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


def _create_stt_provider():
    """
    Pick the best available STT provider. We prefer Deepgram for
    its streaming capabilities, but fall back to local Whisper if
    the API key isn't configured.
    """
    if settings.deepgram_api_key:
        try:
            from src.stt.deepgram_client import DeepgramSTT
            return DeepgramSTT()
        except Exception as exc:
            logger.warning("stt.deepgram_init_failed", error=str(exc))

    # fallback to whisper
    try:
        from src.stt.whisper_client import WhisperSTT
        logger.info("stt.falling_back_to_whisper")
        return WhisperSTT(model_name="base")
    except Exception as exc:
        logger.error("stt.no_provider_available", error=str(exc))
        return None


def _create_tts_provider():
    """
    Same pattern — prefer ElevenLabs, fall back to local pyttsx3.
    """
    if settings.elevenlabs_api_key:
        try:
            from src.tts.elevenlabs_client import ElevenLabsTTS
            return ElevenLabsTTS()
        except Exception as exc:
            logger.warning("tts.elevenlabs_init_failed", error=str(exc))

    try:
        from src.tts.pyttsx_client import LocalTTS
        logger.info("tts.falling_back_to_local")
        return LocalTTS()
    except Exception as exc:
        logger.error("tts.no_provider_available", error=str(exc))
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown logic.
    
    On startup: initialize the vector store, agent, STT, TTS, and
    wire them into the route handlers.
    
    On shutdown: clean up connections.
    """
    logger.info("app.starting", environment=settings.environment)

    # initialize vector store (loads or builds the FAISS index)
    vector_store = get_vector_store()
    logger.info("app.vector_store_ready", docs=vector_store.document_count)

    # initialize providers
    stt = _create_stt_provider()
    tts = _create_tts_provider()
    agent = VoiceAgent()
    metrics = MetricsCollector()
    dashboard = DashboardService(metrics)

    # inject dependencies into route modules
    set_dependencies(agent, metrics)
    set_ws_dependencies(agent, stt, tts, metrics)
    set_twilio_dependencies(agent, stt, tts, metrics)

    # stash on app state so we can access from middleware if needed
    app.state.agent = agent
    app.state.metrics = metrics
    app.state.dashboard = dashboard

    logger.info(
        "app.ready",
        stt_provider=type(stt).__name__ if stt else "none",
        tts_provider=type(tts).__name__ if tts else "none",
    )

    yield

    # shutdown cleanup
    logger.info("app.shutting_down")
    if stt:
        await stt.close()
    if tts:
        await tts.close()


# create the FastAPI app
app = FastAPI(
    title="Real-Time Voice Customer Support Agent",
    description=(
        "AI-powered voice support agent with real-time STT, RAG-based "
        "response generation, TTS, and intelligent escalation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# routes
app.include_router(router)
app.include_router(ws_router)
app.include_router(twilio_router)


# dashboard endpoint lives here because it needs app.state
@app.get("/dashboard")
async def dashboard_endpoint():
    """Full analytics dashboard data in one call."""
    return app.state.dashboard.get_full_dashboard()
