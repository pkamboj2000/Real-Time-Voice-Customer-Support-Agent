# API Documentation

## Base URL

```
http://localhost:8000
```

---

## REST Endpoints

### Health Check

```
GET /health
```

Returns the current health status of the service, including vector store document count.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "vector_store_docs": 21,
  "environment": "development"
}
```

---

### Text Chat

```
POST /chat
```

Send a text message to the agent and receive a response. This is the same pipeline used for voice calls, just without the STT/TTS layers. Useful for testing and integration.

**Request Body:**
```json
{
  "message": "How do I reset my password?",
  "session_id": "optional-session-id",
  "conversation_history": []
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `message` | string | Yes | The user's message |
| `session_id` | string | No | Session identifier for multi-turn conversations |
| `conversation_history` | array | No | Previous turns in `[{"role": "user/assistant", "content": "..."}]` format |

**Response:**
```json
{
  "response": "To reset your password, go to the login page and click 'Forgot Password'...",
  "intent": "account",
  "confidence": 0.92,
  "escalated": false,
  "latency_ms": 1250.5,
  "sources": ["company_faq.json > How do I reset my password?"],
  "session_id": "abc-123"
}
```

---

### Knowledge Base Search

```
POST /search
```

Search the knowledge base directly. Useful for debugging retrieval quality.

**Request Body:**
```json
{
  "query": "pricing plans",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "pricing plans",
  "results": [
    {
      "content": "TechFlow offers three pricing tiers...",
      "title": "What are your pricing plans?",
      "source": "company_faq.json > What are your pricing plans?",
      "score": 0.9312
    }
  ]
}
```

---

### Analytics Metrics

```
GET /metrics
```

Returns aggregate metrics across all recorded interactions.

**Response:**
```json
{
  "total_requests": 150,
  "total_sessions": 42,
  "request_metrics": {
    "avg_latency_ms": 1180.5,
    "min_latency_ms": 450.2,
    "max_latency_ms": 3200.1,
    "escalation_rate": 0.12,
    "resolution_rate": 0.88
  },
  "session_metrics": {
    "avg_turns_per_session": 4.2,
    "avg_session_latency_ms": 1050.3,
    "resolution_rate": 0.85,
    "escalation_rate": 0.15
  },
  "intent_distribution": {
    "billing": 35,
    "account": 28,
    "technical": 42,
    "product_info": 20,
    "cancellation": 8,
    "general": 12,
    "escalation": 5
  },
  "latency_percentiles": {
    "p50": 980.0,
    "p90": 1800.0,
    "p95": 2200.0,
    "p99": 3100.0
  }
}
```

---

### Dashboard

```
GET /dashboard
```

Pre-aggregated dashboard data for frontend consumption.

**Response:**
```json
{
  "overview": {
    "total_calls": 42,
    "total_interactions": 150,
    "resolution_rate": 0.85,
    "escalation_rate": 0.15,
    "avg_response_time_ms": 1180.5,
    "avg_turns_per_call": 4.2
  },
  "latency": {
    "average_ms": 1180.5,
    "min_ms": 450.2,
    "max_ms": 3200.1,
    "percentiles": { "p50": 980.0, "p90": 1800.0, "p95": 2200.0, "p99": 3100.0 }
  },
  "intents": {
    "distribution": { "technical": 42, "billing": 35, "account": 28 },
    "percentages": { "technical": 28.0, "billing": 23.3, "account": 18.7 }
  }
}
```

---

### Escalations

```
GET /escalations
```

View all pending escalations waiting for human agents.

**Response:**
```json
{
  "pending_count": 2,
  "escalations": [
    {
      "session_id": "abc-123",
      "reason": "user_requested",
      "priority": "high",
      "timestamp": "2026-03-15T14:30:00Z",
      "intent": "billing",
      "last_message": "Let me speak to a manager"
    }
  ]
}
```

---

### Rebuild Index

```
POST /admin/rebuild-index
```

Force rebuild the vector store index from knowledge base files. Use after updating documents.

**Response:**
```json
{
  "status": "index_rebuilt",
  "document_count": 21
}
```

---

## WebSocket Endpoints

### Voice Call

```
ws://localhost:8000/ws/call
```

Bidirectional WebSocket for real-time voice communication.

**Client → Server:**
- **Binary frames**: Raw PCM audio (16-bit, 16kHz, mono)
- **Text frames**: JSON control messages
  ```json
  {"type": "end_call"}
  ```

**Server → Client:**
- **Binary frames**: Synthesized audio response (MP3 or WAV)
- **Text frames**: JSON status updates
  ```json
  {"type": "transcript", "role": "caller", "text": "...", "confidence": 0.95}
  {"type": "transcript", "role": "agent", "text": "...", "intent": "billing", "confidence": 0.9}
  {"type": "processing", "step": "transcribing"}
  {"type": "turn_complete", "latency_ms": 1250.5}
  {"type": "call_started", "session_id": "..."}
  {"type": "call_ended", "turns": 5, "resolved": true}
  ```

---

### Twilio Media Stream

```
POST /twilio/voice → TwiML redirect
ws://localhost:8000/twilio/media-stream
```

Handles incoming Twilio phone calls. The POST endpoint returns TwiML that opens a bidirectional media stream. Audio arrives as base64-encoded mulaw at 8kHz in Twilio's JSON message format.

Configure in Twilio console:
1. Set voice webhook URL to `https://your-domain.com/twilio/voice`
2. Ensure your server is publicly accessible (use ngrok for local dev)
