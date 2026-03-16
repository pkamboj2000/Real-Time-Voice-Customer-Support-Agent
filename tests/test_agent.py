"""
Tests for the agent pipeline — intent detection, escalation, and response generation.
"""

import pytest
from unittest.mock import patch, MagicMock

from src.agent.intent import IntentDetector, IntentResult, VALID_INTENTS
from src.agent.escalation import EscalationManager, EscalationReason


class TestIntentParsing:
    """Test the intent classification response parser."""

    def test_parse_standard_format(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("billing|0.92")
        assert intent == "billing"
        assert conf == 0.92

    def test_parse_with_whitespace(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("  technical | 0.85 ")
        assert intent == "technical"
        assert conf == 0.85

    def test_parse_invalid_confidence(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("account|not_a_number")
        assert intent == "account"
        assert conf == 0.5  # default fallback

    def test_parse_no_pipe(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("billing")
        assert intent == "billing"
        assert conf == 0.5

    def test_parse_unknown_intent(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("xyzzy|0.8")
        assert intent == "general"  # falls back to general

    def test_parse_empty(self):
        detector = IntentDetector.__new__(IntentDetector)
        intent, conf = detector._parse_response("")
        assert intent == "general"

    def test_confidence_clamped(self):
        detector = IntentDetector.__new__(IntentDetector)
        _, conf = detector._parse_response("billing|1.5")
        assert conf == 1.0
        _, conf2 = detector._parse_response("billing|-0.3")
        assert conf2 == 0.0


class TestEscalationManager:
    """Test escalation logic and record management."""

    def setup_method(self):
        self.manager = EscalationManager()

    def test_keyword_triggers_escalation(self):
        reason = self.manager.should_escalate(
            session_id="test-1",
            user_message="I want to speak with a manager please",
            intent="general",
            confidence=0.9,
            escalation_keywords=["manager", "human", "supervisor"],
            confidence_threshold=0.65,
        )
        assert reason == EscalationReason.USER_REQUESTED

    def test_low_confidence_triggers_escalation(self):
        reason = self.manager.should_escalate(
            session_id="test-2",
            user_message="something very unusual",
            intent="general",
            confidence=0.3,
            escalation_keywords=["manager"],
            confidence_threshold=0.65,
        )
        assert reason == EscalationReason.LOW_CONFIDENCE

    def test_normal_request_no_escalation(self):
        reason = self.manager.should_escalate(
            session_id="test-3",
            user_message="I need help with my password",
            intent="account",
            confidence=0.9,
            escalation_keywords=["manager"],
            confidence_threshold=0.65,
        )
        assert reason is None

    def test_repeated_failures_trigger_escalation(self):
        # simulate three failures
        for _ in range(3):
            self.manager.record_failure("test-4")

        reason = self.manager.should_escalate(
            session_id="test-4",
            user_message="this is really not working",
            intent="technical",
            confidence=0.85,
            escalation_keywords=["manager"],
            confidence_threshold=0.65,
        )
        assert reason == EscalationReason.REPEATED_FAILURE

    def test_create_and_list_escalations(self):
        self.manager.create_escalation(
            session_id="esc-1",
            reason=EscalationReason.USER_REQUESTED,
            detected_intent="billing",
            confidence_score=0.9,
            last_message="let me talk to someone",
        )

        pending = self.manager.get_pending_escalations()
        assert len(pending) == 1
        assert pending[0].session_id == "esc-1"
        assert pending[0].priority == "high"

    def test_resolve_escalation(self):
        self.manager.create_escalation(
            session_id="esc-2",
            reason=EscalationReason.LOW_CONFIDENCE,
        )
        assert self.manager.get_escalation_count() == 1

        success = self.manager.resolve_escalation("esc-2")
        assert success is True
        assert self.manager.get_escalation_count() == 0

    def test_resolve_nonexistent_returns_false(self):
        assert self.manager.resolve_escalation("nope") is False


class TestIntentNeedsEscalation:
    """Test the needs_escalation helper method."""

    def test_escalation_intent(self):
        detector = IntentDetector.__new__(IntentDetector)
        result = IntentResult(intent="escalation", confidence=0.95)
        assert detector.needs_escalation("I want a supervisor", result) is True

    def test_high_confidence_no_escalation(self):
        detector = IntentDetector.__new__(IntentDetector)
        result = IntentResult(intent="billing", confidence=0.9)
        assert detector.needs_escalation("what's my balance", result) is False
