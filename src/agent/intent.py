"""
Intent detection using LLM-based classification.

We classify caller utterances into intent categories so we can:
  - route to the right knowledge base section
  - decide when to escalate
  - log what kinds of issues people are calling about

Using the LLM for classification instead of a dedicated model
because accuracy matters more than latency here (the classification
runs in parallel with RAG retrieval anyway).
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from openai import OpenAI

from src.agent.prompts import INTENT_CLASSIFICATION_PROMPT
from src.utils.logger import get_logger
from configs.settings import settings

logger = get_logger(__name__)


# valid intent categories — anything outside this set gets mapped to "general"
VALID_INTENTS = {
    "billing", "account", "technical", "product_info",
    "cancellation", "feedback", "escalation", "general",
}


@dataclass
class IntentResult:
    """Detected intent + confidence score."""
    intent: str
    confidence: float
    raw_response: str = ""


class IntentDetector:
    """
    Classifies user messages into intent categories using the LLM.
    """

    def __init__(self):
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_model

    def classify(self, message: str) -> IntentResult:
        """
        Classify a single message. Returns the detected intent
        and confidence score.
        
        The LLM is prompted to return "category|confidence" which
        we parse out. If parsing fails we default to "general" with
        low confidence.
        """
        if not message.strip():
            return IntentResult(intent="general", confidence=0.0)

        prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20,
            )

            raw = response.choices[0].message.content.strip()
            intent, confidence = self._parse_response(raw)

            logger.info(
                "intent.classified",
                message=message[:60],
                intent=intent,
                confidence=confidence,
            )
            return IntentResult(
                intent=intent,
                confidence=confidence,
                raw_response=raw,
            )

        except Exception as exc:
            logger.error("intent.classification_failed", error=str(exc))
            return IntentResult(intent="general", confidence=0.0, raw_response="")

    def _parse_response(self, raw: str) -> Tuple[str, float]:
        """
        Parse the LLM's "category|confidence" response.
        Handles messy outputs gracefully — sometimes the model adds
        extra whitespace or explanations.
        """
        # try the expected format first
        if "|" in raw:
            parts = raw.split("|")
            intent = parts[0].strip().lower()
            try:
                confidence = float(parts[1].strip())
                confidence = max(0.0, min(1.0, confidence))
            except (ValueError, IndexError):
                confidence = 0.5
        else:
            # fallback: just take the first word that looks like an intent
            intent = raw.strip().lower().split()[0] if raw.strip() else "general"
            confidence = 0.5

        # validate the intent category
        if intent not in VALID_INTENTS:
            # see if it's close to a valid one (common typos / variations)
            for valid in VALID_INTENTS:
                if valid in intent or intent in valid:
                    intent = valid
                    break
            else:
                intent = "general"

        return intent, confidence

    def needs_escalation(self, message: str, intent_result: IntentResult) -> bool:
        """
        Check whether this message should trigger escalation to a human.
        We escalate if:
          1. The detected intent is "escalation"
          2. The message contains known escalation keywords
          3. Confidence is too low (the model is unsure)
        """
        if intent_result.intent == "escalation":
            return True

        if intent_result.confidence < settings.confidence_threshold:
            return True

        # check for explicit human-request keywords
        message_lower = message.lower()
        for keyword in settings.escalation_keyword_list:
            if keyword in message_lower:
                return True

        return False
