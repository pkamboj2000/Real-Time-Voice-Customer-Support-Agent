"""
Escalation handler — manages handoff from AI agent to human support.

When the agent decides it can't handle a request (low confidence,
explicit user request, sensitive topic), this module handles the
transition. In production this would integrate with a queueing
system like Genesys or Five9. For now we simulate the handoff
and log the details.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EscalationReason(str, Enum):
    LOW_CONFIDENCE = "low_confidence"
    USER_REQUESTED = "user_requested"
    REPEATED_FAILURE = "repeated_failure"
    SENSITIVE_TOPIC = "sensitive_topic"
    POLICY_VIOLATION = "policy_violation"


@dataclass
class EscalationRecord:
    """Everything a human agent needs to pick up the conversation."""
    session_id: str
    reason: EscalationReason
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    caller_number: Optional[str] = None
    transcript_summary: str = ""
    detected_intent: str = ""
    confidence_score: float = 0.0
    last_caller_message: str = ""
    agent_notes: str = ""
    priority: str = "normal"  # "low", "normal", "high", "urgent"


class EscalationManager:
    """
    Handles the decision-making and logistics of escalating a call
    to a human agent.

    In a real deployment this would push to a queue (redis, SQS, etc.)
    and trigger a notification to the next available human agent.
    Here we log it and store it in memory for the dashboard endpoint.
    """

    def __init__(self):
        # in-memory store — swap for redis or a database in production
        self._pending_escalations: List[EscalationRecord] = []
        self._failure_counts: Dict[str, int] = {}

    def should_escalate(
        self,
        session_id: str,
        user_message: str,
        intent: str,
        confidence: float,
        escalation_keywords: List[str],
        confidence_threshold: float,
    ) -> Optional[EscalationReason]:
        """
        Evaluate whether the current turn should be escalated.
        Returns the reason if yes, None if no.
        """
        # explicit request from the caller
        message_lower = user_message.lower()
        for keyword in escalation_keywords:
            if keyword in message_lower:
                logger.info(
                    "escalation.user_requested",
                    keyword=keyword,
                    session=session_id,
                )
                return EscalationReason.USER_REQUESTED

        # intent classifier flagged it
        if intent == "escalation":
            return EscalationReason.USER_REQUESTED

        # model confidence is too low
        if confidence < confidence_threshold:
            logger.info(
                "escalation.low_confidence",
                confidence=confidence,
                threshold=confidence_threshold,
                session=session_id,
            )
            return EscalationReason.LOW_CONFIDENCE

        # repeated failures — if we've failed to help 3+ times in this session
        failure_count = self._failure_counts.get(session_id, 0)
        if failure_count >= 3:
            return EscalationReason.REPEATED_FAILURE

        return None

    def record_failure(self, session_id: str) -> None:
        """Track when the agent fails to provide a satisfactory response."""
        self._failure_counts[session_id] = self._failure_counts.get(session_id, 0) + 1

    def create_escalation(
        self,
        session_id: str,
        reason: EscalationReason,
        caller_number: Optional[str] = None,
        transcript_summary: str = "",
        detected_intent: str = "",
        confidence_score: float = 0.0,
        last_message: str = "",
    ) -> EscalationRecord:
        """
        Create an escalation record and add it to the queue.
        Returns the record so the caller can use it.
        """
        # priority mapping based on reason
        priority_map = {
            EscalationReason.USER_REQUESTED: "high",
            EscalationReason.LOW_CONFIDENCE: "normal",
            EscalationReason.REPEATED_FAILURE: "high",
            EscalationReason.SENSITIVE_TOPIC: "urgent",
            EscalationReason.POLICY_VIOLATION: "urgent",
        }

        record = EscalationRecord(
            session_id=session_id,
            reason=reason,
            caller_number=caller_number,
            transcript_summary=transcript_summary,
            detected_intent=detected_intent,
            confidence_score=confidence_score,
            last_caller_message=last_message,
            priority=priority_map.get(reason, "normal"),
        )

        self._pending_escalations.append(record)

        logger.info(
            "escalation.created",
            session=session_id,
            reason=reason.value,
            priority=record.priority,
        )

        return record

    def get_pending_escalations(self) -> List[EscalationRecord]:
        """Return all pending escalations (for the dashboard)."""
        return list(self._pending_escalations)

    def resolve_escalation(self, session_id: str) -> bool:
        """Mark an escalation as resolved (human picked it up)."""
        for i, esc in enumerate(self._pending_escalations):
            if esc.session_id == session_id:
                self._pending_escalations.pop(i)
                logger.info("escalation.resolved", session=session_id)
                return True
        return False

    def get_escalation_count(self) -> int:
        return len(self._pending_escalations)
