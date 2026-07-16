"""In-memory session message store (swap later for Redis)."""

from __future__ import annotations

import threading
from collections import defaultdict

from app.config import get_settings

_lock = threading.Lock()
_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)


def get_messages(session_id: str) -> list[dict[str, str]]:
    with _lock:
        return list(_sessions.get(session_id, []))


def append_message(session_id: str, role: str, content: str) -> list[dict[str, str]]:
    cfg = get_settings()
    with _lock:
        msgs = _sessions[session_id]
        msgs.append({"role": role, "content": content})
        # keep last N*2 role messages roughly
        max_msgs = max(4, cfg.session_history_max_turns * 2)
        if len(msgs) > max_msgs:
            _sessions[session_id] = msgs[-max_msgs:]
        return list(_sessions[session_id])


def set_messages(session_id: str, messages: list[dict[str, str]]) -> None:
    with _lock:
        _sessions[session_id] = list(messages)


def clear_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)
