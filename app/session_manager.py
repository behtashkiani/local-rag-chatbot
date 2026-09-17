"""
In-memory session management for short-term conversation history.

Responsibilities
----------------
* Keep a bounded, per-session list of prior messages so the model can answer
  follow-up questions ("what about the airport one?").
* Bound memory two ways: each session keeps only the most recent N messages,
  and the store keeps only the most recent M sessions (LRU eviction). This
  prevents unbounded growth in a long-running process.

This is a single-process store suitable for the brief. For multi-worker or
multi-host deployments, replace the backing dict with Redis behind the same
method surface.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Literal, TypedDict

Role = Literal["user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass
class _Session:
    messages: deque[Message]
    last_access: float = field(default_factory=time.monotonic)


class SessionManager:
    """Thread-safe, bounded conversation history store."""

    def __init__(self, max_history_messages: int = 10, max_sessions: int = 1000):
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self._max_history = max(0, max_history_messages)
        self._max_sessions = max_sessions
        self._sessions: "OrderedDict[str, _Session]" = OrderedDict()
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> list[Message]:
        """Return a copy of the stored history for the session (oldest first).

        Returns an empty list for unknown sessions. Reading also refreshes the
        session's LRU recency.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []
            session.last_access = time.monotonic()
            self._sessions.move_to_end(session_id)
            return list(session.messages)

    def add_exchange(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> None:
        """Append a user/assistant turn pair and enforce all bounds."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                # maxlen keeps only the most recent messages automatically.
                maxlen = self._max_history if self._max_history > 0 else None
                session = _Session(messages=deque(maxlen=maxlen))
                self._sessions[session_id] = session

            if self._max_history > 0:
                session.messages.append({"role": "user", "content": user_message})
                session.messages.append(
                    {"role": "assistant", "content": assistant_message}
                )
            session.last_access = time.monotonic()
            self._sessions.move_to_end(session_id)
            self._evict_if_needed()

    def clear(self, session_id: str) -> bool:
        """Forget a session. Returns True if it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def active_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used sessions beyond the cap. Caller holds lock."""
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
