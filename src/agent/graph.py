"""
LangGraph-based voice support agent — the orchestration layer.

This is the brain of the system. It takes transcribed text from the
caller, runs it through a multi-step pipeline (intent detection →
retrieval → response generation → escalation check), and produces
a text response that gets sent to TTS.

The pipeline is built as a LangGraph StateGraph so each step is
a discrete node. This makes it straightforward to add new steps,
modify the flow, or swap out individual components without touching
the rest.

State machine:
    ┌─────────────┐
    │  classify    │  ← detect intent + confidence
    └──────┬──────┘
           │
    ┌──────▼──────┐      ┌────────────┐
    │  check_esc  │─yes─►│  escalate  │
    └──────┬──────┘      └────────────┘
           │ no
    ┌──────▼──────┐
    │  retrieve   │  ← RAG lookup against knowledge base
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  generate   │  ← LLM response with retrieved context
    └──────┬──────┘
           │
    ┌──────▼──────┐
    │  validate   │  ← sanity-check the response
    └─────────────┘
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from openai import OpenAI

from src.agent.intent import IntentDetector, IntentResult
from src.agent.escalation import EscalationManager, EscalationReason
from src.agent.prompts import (
    SYSTEM_PROMPT,
    RAG_PROMPT_TEMPLATE,
    NO_CONTEXT_PROMPT,
    ESCALATION_PROMPT,
)
from src.retrieval.vector_store import get_vector_store
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# ---- state definition ----

class AgentState(TypedDict):
    """State that flows through the LangGraph pipeline."""
    user_message: str
    session_id: str
    conversation_history: List[Dict[str, str]]

    # populated by the classify node
    intent: str
    confidence: float

    # populated by the escalation check
    should_escalate: bool
    escalation_reason: str

    # populated by the retrieve node
    retrieved_docs: List[Dict[str, Any]]
    retrieval_scores: List[float]

    # populated by the generate node
    agent_response: str

    # timing
    step_latencies: Dict[str, float]
    total_latency_ms: float


# ---- agent class ----

class VoiceAgent:
    """
    The main agent that processes caller messages and returns responses.

    Each call to process_message() runs the full LangGraph pipeline
    and returns an AgentResponse with the text to speak back plus
    metadata about what happened internally.
    """

    def __init__(self):
        self._llm_client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model
        self._intent_detector = IntentDetector()
        self._escalation_manager = EscalationManager()
        self._graph = self._build_graph()

        logger.info("voice_agent.initialized", model=self._model)

    def _build_graph(self) -> StateGraph:
        """
        Construct the LangGraph state machine. Each node is a callable
        that takes the state dict and returns updates to it.
        """
        graph = StateGraph(AgentState)

        # add nodes
        graph.add_node("classify", self._classify_node)
        graph.add_node("check_escalation", self._check_escalation_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("escalate", self._escalate_node)
        graph.add_node("validate", self._validate_node)

        # set entry point
        graph.set_entry_point("classify")

        # edges
        graph.add_edge("classify", "check_escalation")

        # conditional: escalate or continue to retrieval
        graph.add_conditional_edges(
            "check_escalation",
            self._route_after_escalation_check,
            {
                "escalate": "escalate",
                "continue": "retrieve",
            },
        )

        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "validate")
        graph.add_edge("validate", END)
        graph.add_edge("escalate", END)

        return graph.compile()

    def process_message(
        self,
        user_message: str,
        session_id: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single caller message through the full pipeline.
        Returns a dict with the response text and all metadata.

        This is the main entry point that the websocket handler calls.
        """
        start_time = time.monotonic()

        initial_state: AgentState = {
            "user_message": user_message,
            "session_id": session_id,
            "conversation_history": conversation_history or [],
            "intent": "",
            "confidence": 0.0,
            "should_escalate": False,
            "escalation_reason": "",
            "retrieved_docs": [],
            "retrieval_scores": [],
            "agent_response": "",
            "step_latencies": {},
            "total_latency_ms": 0.0,
        }

        # run the graph
        result = self._graph.invoke(initial_state)

        total_ms = (time.monotonic() - start_time) * 1000
        result["total_latency_ms"] = round(total_ms, 2)

        logger.info(
            "agent.message_processed",
            session=session_id,
            intent=result.get("intent"),
            confidence=result.get("confidence"),
            escalated=result.get("should_escalate"),
            latency_ms=round(total_ms, 2),
        )

        return result

    # ---- graph nodes ----

    def _classify_node(self, state: AgentState) -> Dict:
        """Detect the caller's intent and confidence level."""
        start = time.monotonic()

        result = self._intent_detector.classify(state["user_message"])

        elapsed = (time.monotonic() - start) * 1000
        latencies = dict(state.get("step_latencies", {}))
        latencies["classify_ms"] = round(elapsed, 2)

        return {
            "intent": result.intent,
            "confidence": result.confidence,
            "step_latencies": latencies,
        }

    def _check_escalation_node(self, state: AgentState) -> Dict:
        """Decide whether to escalate based on intent and confidence."""
        reason = self._escalation_manager.should_escalate(
            session_id=state["session_id"],
            user_message=state["user_message"],
            intent=state["intent"],
            confidence=state["confidence"],
            escalation_keywords=settings.escalation_keyword_list,
            confidence_threshold=settings.confidence_threshold,
        )

        if reason is not None:
            return {
                "should_escalate": True,
                "escalation_reason": reason.value,
            }
        return {
            "should_escalate": False,
            "escalation_reason": "",
        }

    @staticmethod
    def _route_after_escalation_check(state: AgentState) -> str:
        """Conditional edge: route to escalation or continue."""
        if state.get("should_escalate", False):
            return "escalate"
        return "continue"

    def _retrieve_node(self, state: AgentState) -> Dict:
        """Query the vector store for relevant knowledge base documents."""
        start = time.monotonic()

        store = get_vector_store()
        results = store.search(
            query=state["user_message"],
            top_k=settings.max_retrieval_results,
        )

        docs = []
        scores = []
        for doc, score in results:
            docs.append({
                "content": doc.content,
                "title": doc.title,
                "source": doc.display_source,
                "category": doc.category,
            })
            scores.append(round(float(score), 4))

        elapsed = (time.monotonic() - start) * 1000
        latencies = dict(state.get("step_latencies", {}))
        latencies["retrieve_ms"] = round(elapsed, 2)

        logger.info(
            "agent.retrieval_complete",
            docs_found=len(docs),
            top_score=scores[0] if scores else 0,
        )

        return {
            "retrieved_docs": docs,
            "retrieval_scores": scores,
            "step_latencies": latencies,
        }

    def _generate_node(self, state: AgentState) -> Dict:
        """Generate the agent's response using the LLM with retrieved context."""
        start = time.monotonic()

        # build the context block from retrieved documents
        docs = state.get("retrieved_docs", [])
        if docs:
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.get("source", "unknown")
                content = doc.get("content", "")
                context_parts.append(f"[{i}] ({source}): {content}")

            context = "\n\n".join(context_parts)
            user_prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=state["user_message"],
            )
        else:
            user_prompt = NO_CONTEXT_PROMPT.format(
                question=state["user_message"],
            )

        # build message history for multi-turn context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # add recent conversation history (keep last 10 turns to manage token count)
        history = state.get("conversation_history", [])
        for turn in history[-10:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )
            agent_response = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("agent.generation_failed", error=str(exc))
            agent_response = (
                "I'm sorry, I'm having a bit of trouble right now. "
                "Let me connect you with a team member who can help. "
                "One moment please."
            )
            # mark for manual review
            self._escalation_manager.record_failure(state["session_id"])

        elapsed = (time.monotonic() - start) * 1000
        latencies = dict(state.get("step_latencies", {}))
        latencies["generate_ms"] = round(elapsed, 2)

        return {
            "agent_response": agent_response,
            "step_latencies": latencies,
        }

    def _escalate_node(self, state: AgentState) -> Dict:
        """Generate an escalation message and create the escalation record."""
        start = time.monotonic()

        reason = state.get("escalation_reason", "unknown")
        prompt = ESCALATION_PROMPT.format(
            question=state["user_message"],
            reason=reason,
        )

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=150,
            )
            escalation_message = response.choices[0].message.content.strip()
        except Exception:
            escalation_message = (
                "I completely understand. Let me connect you with one of our "
                "team members right away. They'll have our full conversation "
                "history so you won't need to repeat anything. One moment please."
            )

        # create the escalation record for the human agent queue
        reason_enum = EscalationReason.USER_REQUESTED
        try:
            reason_enum = EscalationReason(reason)
        except ValueError:
            pass

        self._escalation_manager.create_escalation(
            session_id=state["session_id"],
            reason=reason_enum,
            detected_intent=state.get("intent", ""),
            confidence_score=state.get("confidence", 0.0),
            last_message=state["user_message"],
        )

        elapsed = (time.monotonic() - start) * 1000
        latencies = dict(state.get("step_latencies", {}))
        latencies["escalate_ms"] = round(elapsed, 2)

        return {
            "agent_response": escalation_message,
            "step_latencies": latencies,
        }

    def _validate_node(self, state: AgentState) -> Dict:
        """
        Post-generation sanity checks on the response.

        Makes sure we're not sending back anything weird — empty
        responses, responses that are too long for spoken delivery,
        or responses that accidentally include markdown formatting.
        """
        response = state.get("agent_response", "")

        # strip any markdown that might have leaked through
        response = response.replace("**", "")
        response = response.replace("*", "")
        response = response.replace("#", "")
        response = response.replace("`", "")

        # strip bullet point markers (the LLM sometimes can't help itself)
        import re
        response = re.sub(r'^\s*[-•]\s*', '', response, flags=re.MULTILINE)
        response = re.sub(r'^\s*\d+\.\s*', '', response, flags=re.MULTILINE)

        # collapse multiple newlines into spaces (spoken delivery)
        response = re.sub(r'\n+', ' ', response)
        response = re.sub(r'\s+', ' ', response).strip()

        if not response:
            response = (
                "I'm sorry, could you repeat that? "
                "I want to make sure I help you correctly."
            )

        # reasonableness check: if the response is absurdly long,
        # truncate it. ~500 chars is about 30-40 seconds of speech.
        max_chars = 800
        if len(response) > max_chars:
            # try to cut at a sentence boundary
            truncated = response[:max_chars]
            last_period = truncated.rfind(".")
            if last_period > max_chars * 0.6:
                response = truncated[:last_period + 1]
            else:
                response = truncated.rstrip() + "..."

        return {"agent_response": response}

    # ---- public helpers ----

    @property
    def escalation_manager(self) -> EscalationManager:
        """Expose the escalation manager for the API layer."""
        return self._escalation_manager
