"""
Evaluation script — measures response quality and latency.

Runs a set of predefined test cases through the agent pipeline and
evaluates:
  1. Latency — end-to-end response time for each query
  2. Grounding — whether the response references retrieved documents
  3. Escalation accuracy — correct escalation decisions
  4. Response quality — basic checks (non-empty, reasonable length, etc.)

Usage:
    python scripts/evaluate.py
    # or
    make evaluate
"""

import sys
import os
import json
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# evaluation test cases — each has an input, expected intent, and
# whether escalation is expected
EVAL_CASES = [
    {
        "message": "How do I reset my password?",
        "expected_intent": "account",
        "expect_escalation": False,
        "should_reference": ["password", "reset", "login"],
    },
    {
        "message": "What are your pricing plans?",
        "expected_intent": "product_info",
        "expect_escalation": False,
        "should_reference": ["starter", "professional", "enterprise"],
    },
    {
        "message": "I want to cancel my subscription immediately",
        "expected_intent": "cancellation",
        "expect_escalation": False,
        "should_reference": ["cancel", "billing", "settings"],
    },
    {
        "message": "Let me speak with your manager right now",
        "expected_intent": "escalation",
        "expect_escalation": True,
        "should_reference": ["connect", "team"],
    },
    {
        "message": "The app keeps crashing when I try to upload files",
        "expected_intent": "technical",
        "expect_escalation": False,
        "should_reference": ["upload", "file"],
    },
    {
        "message": "Why was I charged $50 when my plan is $29?",
        "expected_intent": "billing",
        "expect_escalation": False,
        "should_reference": ["charge", "invoice", "billing"],
    },
    {
        "message": "asdlkfjaslkdfj",
        "expected_intent": "general",
        "expect_escalation": True,  # gibberish should trigger low confidence
        "should_reference": [],
    },
    {
        "message": "Is my data encrypted and secure?",
        "expected_intent": "product_info",
        "expect_escalation": False,
        "should_reference": ["encrypt", "secure", "SOC"],
    },
]


def evaluate_grounding(response: str, expected_terms: List[str]) -> float:
    """
    Check what fraction of expected terms appear in the response.
    This is a rough measure of whether the response is grounded in
    the knowledge base rather than hallucinated.
    """
    if not expected_terms:
        return 1.0  # nothing to check

    response_lower = response.lower()
    found = sum(1 for term in expected_terms if term.lower() in response_lower)
    return found / len(expected_terms)


def evaluate_response_quality(response: str) -> Dict[str, bool]:
    """Basic response quality checks."""
    return {
        "non_empty": len(response.strip()) > 0,
        "reasonable_length": 10 < len(response) < 1000,
        "no_markdown": "**" not in response and "```" not in response,
        "no_bullet_points": not any(response.strip().startswith(c) for c in ["- ", "* ", "1."]),
        "ends_naturally": response.rstrip()[-1] in ".?!" if response.strip() else False,
    }


def main():
    from src.agent.graph import VoiceAgent
    from src.retrieval.vector_store import get_vector_store

    print("=" * 60)
    print("  Voice Agent Evaluation Suite")
    print("=" * 60)

    # make sure the vector store is loaded
    print("\nLoading vector store...")
    store = get_vector_store()
    print(f"Vector store ready with {store.document_count} documents")

    # initialize agent
    print("Initializing agent...\n")
    agent = VoiceAgent()

    # run evaluation
    results = []
    total_latency = 0
    intent_correct = 0
    escalation_correct = 0
    total_grounding = 0

    for i, case in enumerate(EVAL_CASES, 1):
        print(f"--- Test Case {i}/{len(EVAL_CASES)} ---")
        print(f"Input: {case['message']}")

        start = time.monotonic()
        result = agent.process_message(
            user_message=case["message"],
            session_id=f"eval-{i}",
        )
        latency = (time.monotonic() - start) * 1000

        response = result.get("agent_response", "")
        intent = result.get("intent", "")
        confidence = result.get("confidence", 0.0)
        escalated = result.get("should_escalate", False)

        # intent accuracy
        intent_match = intent == case["expected_intent"]
        if intent_match:
            intent_correct += 1

        # escalation accuracy
        esc_match = escalated == case["expect_escalation"]
        if esc_match:
            escalation_correct += 1

        # grounding score
        grounding = evaluate_grounding(response, case["should_reference"])
        total_grounding += grounding

        # quality checks
        quality = evaluate_response_quality(response)

        total_latency += latency

        print(f"Response: {response[:120]}...")
        print(f"Intent: {intent} (expected: {case['expected_intent']}) {'OK' if intent_match else 'MISMATCH'}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Escalated: {escalated} (expected: {case['expect_escalation']}) {'OK' if esc_match else 'MISMATCH'}")
        print(f"Grounding: {grounding:.0%}")
        print(f"Latency: {latency:.0f}ms")
        print(f"Quality: {quality}")
        print()

        results.append({
            "case": i,
            "message": case["message"],
            "intent": intent,
            "expected_intent": case["expected_intent"],
            "intent_correct": intent_match,
            "escalated": escalated,
            "escalation_correct": esc_match,
            "grounding_score": grounding,
            "latency_ms": round(latency, 2),
            "quality": quality,
        })

    # summary
    n = len(EVAL_CASES)
    print("=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total test cases:       {n}")
    print(f"  Intent accuracy:        {intent_correct}/{n} ({intent_correct/n:.0%})")
    print(f"  Escalation accuracy:    {escalation_correct}/{n} ({escalation_correct/n:.0%})")
    print(f"  Avg grounding score:    {total_grounding/n:.0%}")
    print(f"  Avg latency:            {total_latency/n:.0f}ms")
    print(f"  Total latency:          {total_latency:.0f}ms")
    print(f"  Min latency:            {min(r['latency_ms'] for r in results):.0f}ms")
    print(f"  Max latency:            {max(r['latency_ms'] for r in results):.0f}ms")
    print("=" * 60)

    # save detailed results
    os.makedirs("logs", exist_ok=True)
    output_path = "logs/evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "total_cases": n,
                "intent_accuracy": round(intent_correct / n, 4),
                "escalation_accuracy": round(escalation_correct / n, 4),
                "avg_grounding_score": round(total_grounding / n, 4),
                "avg_latency_ms": round(total_latency / n, 2),
            },
            "results": results,
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
