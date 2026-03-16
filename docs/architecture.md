# Architecture

## System Overview

The Real-Time Voice Customer Support Agent is a modular pipeline that handles inbound customer support calls end-to-end. Audio comes in from the caller, gets transcribed, processed by an intelligent agent, and the response is synthesized back into speech — all in real time over a single WebSocket connection.

## Architecture Diagram

```
┌──────────────┐     ┌───────────────────────────────────────────────────────────────┐
│              │     │                    FastAPI Server                              │
│   Caller     │     │                                                               │
│  (Phone)     │     │  ┌─────────────┐    ┌──────────────────────────────────────┐  │
│              │     │  │             │    │       LangGraph Agent Pipeline       │  │
│              │     │  │   STT       │    │                                      │  │
│    ┌─────┐   │ WS  │  │  Module    │    │  ┌──────────┐    ┌──────────────┐   │  │
│    │     │───┼─────┼──►           ├────►  │ Intent   │    │  Escalation  │   │  │
│    │     │   │Audio│  │ Deepgram   │Text│  │ Classify │───►│  Check       │   │  │
│    │     │   │     │  │ / Whisper  │    │  └─────┬────┘    └──────┬───────┘   │  │
│    │     │   │     │  │             │    │        │               │           │  │
│    │     │   │     │  └─────────────┘    │        │ yes           │ no        │  │
│    │     │   │     │                     │        ▼               ▼           │  │
│    │     │   │     │  ┌─────────────┐    │  ┌──────────┐    ┌──────────┐     │  │
│    │     │◄──┼─────┼──┤             │◄───┤  │ Escalate │    │ RAG      │     │  │
│    │     │   │Audio│  │   TTS       │Text│  │ Handler  │    │ Retrieve │     │  │
│    └─────┘   │     │  │  Module     │    │  └──────────┘    └─────┬────┘     │  │
│              │     │  │ ElevenLabs  │    │                        │          │  │
│              │     │  │ / pyttsx3   │    │                  ┌─────▼────┐     │  │
│              │     │  └─────────────┘    │                  │ LLM      │     │  │
│              │     │                     │                  │ Generate  │     │  │
│              │     │                     │                  └─────┬────┘     │  │
│              │     │                     │                        │          │  │
│              │     │                     │                  ┌─────▼────┐     │  │
│              │     │                     │                  │ Validate │     │  │
│              │     │                     │                  └──────────┘     │  │
│              │     │                     └──────────────────────────────────────┘  │
│              │     │                                                               │
│              │     │  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐     │
│              │     │  │  Transcript │    │   Metrics    │    │  FAISS       │     │
│              │     │  │  Store      │    │   Collector  │    │  Vector DB   │     │
│              │     │  └─────────────┘    └──────────────┘    └──────────────┘     │
└──────────────┘     └───────────────────────────────────────────────────────────────┘
                              │                     │
                     ┌────────▼────────┐   ┌────────▼────────┐
                     │  Twilio PSTN    │   │  Analytics      │
                     │  Gateway        │   │  Dashboard      │
                     └─────────────────┘   └─────────────────┘
```

## Pipeline Flow

### 1. Audio Ingestion
- Caller connects via WebSocket (direct) or Twilio Media Streams (phone)
- Raw audio arrives as 16-bit PCM at 16kHz mono
- Audio is buffered with voice activity detection (VAD) to determine when the caller stops speaking

### 2. Speech-to-Text (STT)
- **Primary**: Deepgram Nova-2 streaming API — low-latency, high accuracy
- **Fallback**: Local Whisper model — works offline, no API key needed
- Produces interim and final transcription results
- Confidence scores are passed downstream for escalation decisions

### 3. Agent Pipeline (LangGraph)
The agent pipeline is built as a LangGraph StateGraph with five nodes:

| Node | Purpose |
|------|---------|
| `classify` | Detect caller intent using LLM-based classification |
| `check_escalation` | Evaluate whether to hand off to a human agent |
| `retrieve` | Query FAISS vector store for relevant knowledge base documents |
| `generate` | Generate a grounded response using the LLM with retrieved context |
| `validate` | Post-process the response (strip markdown, length check, etc.) |

The pipeline conditionally routes to an `escalate` node if:
- The caller explicitly asks for a human
- Intent confidence is below the configured threshold
- The message contains escalation keywords
- The agent has failed to help 3+ times in the session

### 4. Text-to-Speech (TTS)
- **Primary**: ElevenLabs multilingual v2 — high-quality neural voice
- **Fallback**: pyttsx3 — local engine, works offline
- Audio is streamed back to the caller over the WebSocket

### 5. Barge-In Handling
- While the agent is speaking, incoming audio is monitored
- If the caller starts talking (detected via energy-based VAD), agent playback is interrupted
- The new caller utterance is processed normally

## Data Flow

```
[Audio In] → [VAD Filter] → [STT] → [Intent Classification]
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                        [Escalation?]            [RAG Query]
                              │                       │
                        [Human Queue]           [LLM Generate]
                                                      │
                                                [Validate]
                                                      │
                                                [TTS] → [Audio Out]
```

## Key Design Decisions

1. **LangGraph over plain LangChain**: The state machine approach gives us explicit control over the flow, makes it easy to add new steps, and provides better observability.

2. **Provider abstraction**: Both STT and TTS have abstract base classes so we can swap providers without modifying the pipeline.

3. **Graceful degradation**: If a cloud provider is down or unconfigured, the system automatically falls back to local alternatives.

4. **Dual embedding support**: OpenAI embeddings for production quality, sentence-transformers for offline/free operation.

5. **In-process vector store**: FAISS runs in-process for minimum latency. No separate database to manage.

## Metrics and Observability

Every request records:
- STT transcription latency
- Intent classification result and confidence
- RAG retrieval scores
- LLM generation latency
- TTS synthesis latency
- Total end-to-end latency
- Escalation decisions
- Resolution outcomes

These flow into the `MetricsCollector` which provides aggregate analytics through the `/metrics` and `/dashboard` endpoints.
