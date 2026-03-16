# Real-Time Voice Customer Support Agent

An end-to-end voice AI system that handles customer support phone calls in real time. Accepts live audio, transcribes speech incrementally, performs intent-aware retrieval-augmented generation against a company knowledge base, synthesizes spoken responses, and intelligently escalates to human agents when needed.

Built with FastAPI, LangGraph, Deepgram, ElevenLabs, and FAISS.

---

## Architecture

```
Caller ──► Twilio/WebSocket ──► STT (Deepgram) ──► LangGraph Agent ──► TTS (ElevenLabs) ──► Caller
                                                         │
                                                    ┌────┴────┐
                                                    │         │
                                              RAG Retrieval  Escalation
                                              (FAISS + LLM)  (Human Queue)
```

The agent pipeline is a LangGraph state machine with five stages:

| Stage | What it does |
|-------|-------------|
| **Classify** | Detects caller intent (billing, technical, account, etc.) with confidence scoring |
| **Escalation Check** | Routes to human agent if confidence is low, user requests it, or after repeated failures |
| **Retrieve** | Queries FAISS vector store over company docs and FAQs for relevant context |
| **Generate** | Produces a grounded, conversational response using GPT-4o with retrieved context |
| **Validate** | Strips markdown, enforces length limits, ensures the response sounds natural when spoken |

Full architecture details: [docs/architecture.md](docs/architecture.md)

---

## Features

- **Real-time voice pipeline** — WebSocket-based audio streaming with sub-2-second end-to-end latency
- **Dual STT providers** — Deepgram streaming (primary) with automatic Whisper fallback
- **Dual TTS providers** — ElevenLabs neural voices (primary) with pyttsx3 local fallback
- **RAG over company knowledge base** — FAISS vector search with OpenAI or sentence-transformer embeddings
- **Intent classification** — LLM-based intent detection with 8 categories and confidence scoring
- **Smart escalation** — Hands off to humans based on confidence threshold, explicit requests, and repeated failures
- **Barge-in detection** — Interruption handling when the caller starts talking over the agent
- **Twilio telephony** — Ready-to-use Twilio Media Streams integration for real phone numbers
- **Transcript logging** — Every call is saved with full transcript, timestamps, intents, and latency data
- **Analytics dashboard** — Resolution rate, latency percentiles, intent distribution, escalation metrics
- **Evaluation suite** — Automated testing for response grounding, intent accuracy, and latency benchmarks
- **Graceful degradation** — If any cloud provider fails, the system falls back to local alternatives

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend framework | FastAPI + Uvicorn |
| Real-time communication | WebSockets |
| Speech-to-text | Deepgram Nova-2, OpenAI Whisper |
| Text-to-speech | ElevenLabs, pyttsx3 |
| Agent orchestration | LangGraph (state machine) |
| LLM | OpenAI GPT-4o |
| Retrieval | FAISS vector store |
| Embeddings | OpenAI text-embedding-3-small / sentence-transformers |
| Telephony | Twilio Media Streams |
| Config | pydantic-settings + .env |
| Logging | structlog (JSON in prod, console in dev) |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
├── src/
│   ├── main.py                    # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py              # REST endpoints (chat, search, metrics)
│   │   ├── websocket_handler.py   # Real-time voice WebSocket
│   │   └── middleware.py          # Request logging
│   ├── stt/
│   │   ├── base.py                # STT provider interface
│   │   ├── deepgram_client.py     # Deepgram streaming STT
│   │   └── whisper_client.py      # Local Whisper fallback
│   ├── tts/
│   │   ├── base.py                # TTS provider interface
│   │   ├── elevenlabs_client.py   # ElevenLabs neural TTS
│   │   └── pyttsx_client.py       # Local pyttsx3 fallback
│   ├── agent/
│   │   ├── graph.py               # LangGraph agent pipeline
│   │   ├── intent.py              # Intent classification
│   │   ├── escalation.py          # Escalation logic and queue
│   │   └── prompts.py             # System prompts and templates
│   ├── retrieval/
│   │   ├── vector_store.py        # FAISS index management
│   │   ├── embeddings.py          # Embedding generation (OpenAI / local)
│   │   └── knowledge_loader.py    # Knowledge base document loader
│   ├── telephony/
│   │   └── twilio_handler.py      # Twilio Media Streams integration
│   ├── analytics/
│   │   ├── metrics.py             # Metrics collection
│   │   └── dashboard.py           # Dashboard data service
│   └── utils/
│       ├── logger.py              # Structured logging setup
│       ├── audio.py               # Audio processing utilities
│       └── transcript.py          # Transcript storage
├── knowledge_base/                 # Company docs (FAQs, product info, policies)
├── scripts/
│   ├── build_index.py             # Build FAISS index from knowledge base
│   ├── evaluate.py                # Evaluation suite (grounding, latency, accuracy)
│   └── simulate_call.py           # Multi-turn call simulation
├── samples/                        # Sample call flow definitions
├── tests/                          # Pytest test suite
├── docs/
│   ├── architecture.md            # Detailed architecture documentation
│   └── api.md                     # API endpoint documentation
├── configs/
│   └── settings.py                # Pydantic settings management
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── .env.example
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenAI API key (required for the LLM and embeddings)
- Deepgram API key (optional — falls back to Whisper)
- ElevenLabs API key (optional — falls back to pyttsx3)

### 1. Clone and install

```bash
git clone https://github.com/pkamboj2000/Real-Time-Voice-Customer-Support-Agent.git
cd Real-Time-Voice-Customer-Support-Agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

At minimum you need `OPENAI_API_KEY`. The STT and TTS providers will fall back to local alternatives if their keys aren't set.

### 3. Build the vector index

```bash
python scripts/build_index.py
```

This reads the knowledge base files from `knowledge_base/`, generates embeddings, and saves the FAISS index to `data/vector_index/`.

Expected output:
```
============================================================
  Building vector index from knowledge base
============================================================

Loaded 21 document chunks from knowledge base
  company_faq.json: 12 chunks
  policies.json: 4 chunks
  product_docs.json: 5 chunks

Generating embeddings and building index...

Index built and saved to ./data/vector_index/
Total documents indexed: 21
Time taken: 3.2 seconds

--- Sanity check ---
  Q: How do I reset my password?
  A: [0.891] How do I reset my password? (company_faq.json)

  Q: What are your pricing plans?
  A: [0.923] What are your pricing plans? (company_faq.json)

Done.
```

### 4. Run the server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Make:
```bash
make run
```

### 5. Test it

Text-based chat (no audio needed):
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?"}'
```

Simulate a full call:
```bash
python scripts/simulate_call.py --flow 1
```

Run the evaluation suite:
```bash
python scripts/evaluate.py
```

### 6. Docker (optional)

```bash
docker-compose up --build
```

---

## Sample Call Simulation Output

```
============================================================
  Simulating: Account Password Reset
  Scenario: Caller needs help resetting their password
============================================================

  Caller:  Hi, I'm locked out of my account and I can't remember my password. Can you help?
  Agent:   Of course, I'd be happy to help you get back into your account. You can reset your
           password by going to the login page and clicking 'Forgot Password'. Just enter the
           email address on your account and you'll get a reset link within about five minutes.
           Is there anything else I can help with?
           [intent=account, conf=0.94, latency=1180ms]

  Caller:  I checked my spam folder already. Still nothing.
  Agent:   I understand how frustrating that can be. If the reset email isn't showing up in your
           inbox or spam folder, it's possible the email address on file might be different from
           what you're checking. Would you like me to connect you with our account team? They
           can verify the email on your account and send the reset link manually.
           [intent=account, conf=0.89, latency=1350ms]
```

---

## API Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Text-based agent chat |
| `/search` | POST | Knowledge base search |
| `/metrics` | GET | Analytics metrics |
| `/dashboard` | GET | Dashboard data |
| `/escalations` | GET | Pending human escalations |
| `/admin/rebuild-index` | POST | Rebuild vector index |
| `/ws/call` | WebSocket | Real-time voice call |
| `/twilio/voice` | POST | Twilio incoming call webhook |
| `/twilio/media-stream` | WebSocket | Twilio media stream |

Full API documentation: [docs/api.md](docs/api.md)

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Evaluation

The evaluation script tests the agent against predefined test cases and measures:

- **Intent accuracy** — correct intent classification across categories
- **Escalation accuracy** — appropriate escalation/non-escalation decisions
- **Grounding score** — whether responses reference knowledge base content
- **Latency** — end-to-end response time per query

```bash
python scripts/evaluate.py
```

```
============================================================
  EVALUATION SUMMARY
============================================================
  Total test cases:       8
  Intent accuracy:        7/8 (88%)
  Escalation accuracy:    8/8 (100%)
  Avg grounding score:    82%
  Avg latency:            1240ms
============================================================
```

---



