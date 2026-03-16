"""
Call flow simulator — tests the full pipeline with predefined conversations.

Loads sample call flows from samples/ and runs them through the agent
pipeline step by step, printing the interaction like a phone call
transcript. Useful for demos and for verifying the agent handles
multi-turn conversations correctly.

Usage:
    python scripts/simulate_call.py
    python scripts/simulate_call.py --flow 2
    # or
    make simulate
"""

import sys
import os
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_call_flow(flow_number: int) -> dict:
    """Load a sample call flow from the samples/ directory."""
    filepath = f"samples/sample_call_flow_{flow_number}.json"
    if not os.path.exists(filepath):
        print(f"Call flow file not found: {filepath}")
        sys.exit(1)

    with open(filepath, "r") as f:
        return json.load(f)


def simulate(flow_number: int):
    from src.agent.graph import VoiceAgent
    from src.retrieval.vector_store import get_vector_store
    from src.utils.transcript import TranscriptStore

    # load vector store
    print("Loading knowledge base...")
    store = get_vector_store()
    print(f"Ready. {store.document_count} documents indexed.\n")

    # load call flow
    flow = load_call_flow(flow_number)
    print(f"{'=' * 60}")
    print(f"  Simulating: {flow['title']}")
    print(f"  Scenario: {flow['description']}")
    print(f"{'=' * 60}\n")

    # initialize
    agent = VoiceAgent()
    transcript = TranscriptStore(caller_number=flow.get("caller_number", "+1555000000"))
    conversation_history = []
    session_id = f"sim-{flow_number}-{int(time.time())}"

    # walk through each turn
    for turn in flow["turns"]:
        caller_text = turn["caller"]

        print(f"  Caller:  {caller_text}")
        transcript.add_caller_turn(caller_text)
        conversation_history.append({"role": "user", "content": caller_text})

        # process through agent
        start = time.monotonic()
        result = agent.process_message(
            user_message=caller_text,
            session_id=session_id,
            conversation_history=conversation_history,
        )
        elapsed = (time.monotonic() - start) * 1000

        agent_response = result.get("agent_response", "")
        intent = result.get("intent", "")
        confidence = result.get("confidence", 0.0)
        escalated = result.get("should_escalate", False)

        print(f"  Agent:   {agent_response}")
        print(f"           [intent={intent}, conf={confidence:.2f}, latency={elapsed:.0f}ms]")

        if escalated:
            print(f"           ** ESCALATED — transferring to human agent **")

        print()

        transcript.add_agent_turn(
            text=agent_response,
            intent=intent,
            confidence=confidence,
            latency_ms=elapsed,
            escalated=escalated,
        )
        conversation_history.append({"role": "assistant", "content": agent_response})

        # small pause for readability
        time.sleep(0.2)

    # finalize
    resolved = not transcript.session.escalated
    data = transcript.finalize(resolved=resolved)
    filepath = transcript.save_to_disk()

    print(f"{'=' * 60}")
    print(f"  Call Summary")
    print(f"{'=' * 60}")
    print(f"  Session ID:    {transcript.session_id}")
    print(f"  Total turns:   {data['turn_count']}")
    print(f"  Avg latency:   {data['avg_latency_ms']:.0f}ms")
    print(f"  Resolved:      {data['resolved']}")
    print(f"  Escalated:     {data['escalated']}")
    print(f"  Transcript:    {filepath}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Simulate a customer support call")
    parser.add_argument(
        "--flow", type=int, default=1,
        help="Call flow number to simulate (1, 2, or 3)"
    )
    args = parser.parse_args()
    simulate(args.flow)


if __name__ == "__main__":
    main()
