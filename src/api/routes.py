"""
REST API routes — health checks, text-based chat, analytics, and admin endpoints.

The real-time voice path goes through the WebSocket handler, but these
REST endpoints are useful for:
  - health checks (load balancers, Docker)
  - text-based testing (no audio needed)
  - analytics dashboard data
  - admin operations (rebuild index, view escalations)
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agent.graph import VoiceAgent
from src.retrieval.vector_store import get_vector_store
from src.analytics.metrics import MetricsCollector
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

# these are initialized in main.py and injected here
_agent: Optional[VoiceAgent] = None
_metrics: Optional[MetricsCollector] = None


def set_dependencies(agent: VoiceAgent, metrics: MetricsCollector):
    """Called from main.py during startup to inject shared instances."""
    global _agent, _metrics
    _agent = agent
    _metrics = metrics


# ---- request/response models ----

class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message text")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_history: List[Dict] = Field(
        default_factory=list,
        description="Previous conversation turns"
    )


class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    escalated: bool
    latency_ms: float
    sources: List[str]
    session_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_docs: int
    environment: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for knowledge base")
    top_k: int = Field(5, description="Number of results to return")


# ---- routes ----

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health endpoint — used by Docker healthcheck and load balancers."""
    store = get_vector_store()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_docs=store.document_count,
        environment="development",
    )


@router.post("/chat", response_model=ChatResponse)
async def text_chat(req: ChatRequest):
    """
    Text-based chat endpoint for testing the agent without audio.
    Same pipeline as the voice path, just without STT/TTS.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized yet")

    import uuid
    session_id = req.session_id or str(uuid.uuid4())

    result = _agent.process_message(
        user_message=req.message,
        session_id=session_id,
        conversation_history=req.conversation_history,
    )

    # extract source citations from retrieved docs
    sources = []
    for doc in result.get("retrieved_docs", []):
        src = doc.get("source", "")
        if src and src not in sources:
            sources.append(src)

    # record metrics
    if _metrics is not None:
        _metrics.record_request(
            session_id=session_id,
            intent=result.get("intent", ""),
            latency_ms=result.get("total_latency_ms", 0),
            escalated=result.get("should_escalate", False),
            resolved=not result.get("should_escalate", False),
        )

    return ChatResponse(
        response=result.get("agent_response", ""),
        intent=result.get("intent", ""),
        confidence=result.get("confidence", 0.0),
        escalated=result.get("should_escalate", False),
        latency_ms=result.get("total_latency_ms", 0.0),
        sources=sources,
        session_id=session_id,
    )


@router.post("/search")
async def search_knowledge_base(req: SearchRequest):
    """
    Direct search against the knowledge base — useful for debugging
    retrieval quality or building a search interface.
    """
    store = get_vector_store()
    results = store.search(query=req.query, top_k=req.top_k)

    return {
        "query": req.query,
        "results": [
            {
                "content": doc.content,
                "title": doc.title,
                "source": doc.display_source,
                "score": round(score, 4),
            }
            for doc, score in results
        ],
    }


@router.get("/escalations")
async def get_escalations():
    """List all pending escalations for the human agent dashboard."""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    escalations = _agent.escalation_manager.get_pending_escalations()
    return {
        "pending_count": len(escalations),
        "escalations": [
            {
                "session_id": esc.session_id,
                "reason": esc.reason.value if hasattr(esc.reason, "value") else esc.reason,
                "priority": esc.priority,
                "timestamp": esc.timestamp,
                "intent": esc.detected_intent,
                "last_message": esc.last_caller_message,
            }
            for esc in escalations
        ],
    }


@router.get("/metrics")
async def get_metrics():
    """Return analytics metrics for the dashboard."""
    if _metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")

    return _metrics.get_summary()


@router.post("/admin/rebuild-index")
async def rebuild_index():
    """
    Force rebuild the vector store index from knowledge base files.
    Use this after updating the knowledge base content.
    """
    from src.retrieval.knowledge_loader import KnowledgeBaseLoader

    loader = KnowledgeBaseLoader()
    documents = loader.load_all()

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found in knowledge base")

    store = get_vector_store()
    store.build_index(documents)
    store.save()

    return {
        "status": "index_rebuilt",
        "document_count": len(documents),
    }
