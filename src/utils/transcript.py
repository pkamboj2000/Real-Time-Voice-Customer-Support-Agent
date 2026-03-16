"""
Transcript storage — keeps a running record of each call session.

Each call session gets its own TranscriptStore instance. After the call
ends we dump the full transcript (with timestamps and metadata) either
to disk or to whatever persistence layer we hook up later.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptEntry:
    """Single turn in a conversation — either caller or agent."""
    role: str                 # "caller" or "agent"
    text: str
    timestamp: float          # unix epoch
    intent: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    escalated: bool = False
    retrieval_sources: List[str] = field(default_factory=list)


@dataclass
class CallSession:
    """All data associated with one phone call / websocket session."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    ended_at: Optional[str] = None
    caller_number: Optional[str] = None
    entries: List[TranscriptEntry] = field(default_factory=list)
    resolved: bool = False
    escalated: bool = False
    total_latency_ms: float = 0.0

    # aggregate metrics filled in at the end
    turn_count: int = 0
    avg_latency_ms: float = 0.0
    fallback_count: int = 0


class TranscriptStore:
    """
    Manages transcript data for a single call. Create one per session.

    Usage:
        store = TranscriptStore()
        store.add_caller_turn("I need help with my order")
        store.add_agent_turn("Sure, let me look that up.", intent="order_status", ...)
        store.finalize(resolved=True)
        store.save_to_disk()
    """

    def __init__(self, caller_number: Optional[str] = None):
        self.session = CallSession(caller_number=caller_number)
        self._turn_latencies: List[float] = []

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def add_caller_turn(self, text: str) -> None:
        entry = TranscriptEntry(
            role="caller",
            text=text,
            timestamp=time.time(),
        )
        self.session.entries.append(entry)
        logger.info("transcript.caller_turn", text=text[:80], session=self.session_id)

    def add_agent_turn(
        self,
        text: str,
        intent: Optional[str] = None,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        escalated: bool = False,
        retrieval_sources: Optional[List[str]] = None,
    ) -> None:
        entry = TranscriptEntry(
            role="agent",
            text=text,
            timestamp=time.time(),
            intent=intent,
            confidence=confidence,
            latency_ms=latency_ms,
            escalated=escalated,
            retrieval_sources=retrieval_sources or [],
        )
        self.session.entries.append(entry)

        if latency_ms is not None:
            self._turn_latencies.append(latency_ms)

        if escalated:
            self.session.escalated = True
            self.session.fallback_count += 1

        logger.info(
            "transcript.agent_turn",
            intent=intent,
            confidence=confidence,
            latency_ms=latency_ms,
            session=self.session_id,
        )

    def finalize(self, resolved: bool = False) -> Dict:
        """
        Mark the session as complete and compute aggregate stats.
        Returns the full session dict for convenience.
        """
        self.session.ended_at = datetime.now(timezone.utc).isoformat()
        self.session.resolved = resolved
        self.session.turn_count = len(self.session.entries)

        if self._turn_latencies:
            self.session.avg_latency_ms = round(
                sum(self._turn_latencies) / len(self._turn_latencies), 2
            )
            self.session.total_latency_ms = round(sum(self._turn_latencies), 2)

        logger.info(
            "transcript.finalized",
            session=self.session_id,
            turns=self.session.turn_count,
            resolved=resolved,
            avg_latency=self.session.avg_latency_ms,
        )
        return self.to_dict()

    def to_dict(self) -> Dict:
        return asdict(self.session)

    def save_to_disk(self, directory: str = "logs/transcripts") -> str:
        """
        Persist the transcript as a JSON file. Returns the file path.
        We use the session ID as the filename so it's easy to correlate
        with log entries.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(directory, f"{self.session_id}.json")

        with open(filepath, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

        logger.info("transcript.saved", path=filepath)
        return filepath
