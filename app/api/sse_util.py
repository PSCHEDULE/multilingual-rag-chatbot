"""SSE body parsing and smoke-contract validation (shared by tests and smoke script)."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_sse_events(body: str) -> list[dict[str, Any]]:
    """
    Parse SSE text into ordered events: ``{"event": str, "data": dict|str}``.

    Ignores comment/keepalive lines (e.g. ``: ping``).
    """
    events: list[dict[str, Any]] = []
    # Normalize newlines; split on blank lines
    text = (body or "").replace("\r\n", "\n")
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue  # keepalive / comment
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines and event_name == "message":
            continue
        raw = "\n".join(data_lines)
        data: Any
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = raw
        events.append({"event": event_name, "data": data})
    return events


def validate_smoke_sse_body(body: str) -> None:
    """
    Validate a full SSE response for developer smoke checks.

    Requires:
    - at least one ``meta`` event
    - at least one ``sources`` event
    - a terminal ``done`` event
    - no ``error`` event
    - ``meta`` appears before any ``token`` / ``sources`` / ``done``
    - ``done`` is the last event

    Raises ``ValueError`` with a short reason on failure.
    """
    events = parse_sse_events(body)
    if not events:
        raise ValueError("no SSE events parsed from response body")

    names = [str(e.get("event") or "message") for e in events]
    if "error" in names:
        raise ValueError("error event present in SSE stream")
    if "meta" not in names:
        raise ValueError("missing required meta event")
    if "sources" not in names:
        raise ValueError("missing required sources event")
    if "done" not in names:
        raise ValueError("missing required done event")

    if names[0] != "meta":
        raise ValueError(f"meta must be first event, got {names[0]!r}")
    if names[-1] != "done":
        raise ValueError(f"done must be terminal event, got {names[-1]!r}")

    # Logical order: meta before token/sources/done (already first); done last
    first_meta = names.index("meta")
    for later in ("token", "sources", "done"):
        if later in names and names.index(later) < first_meta:
            raise ValueError(f"{later} appeared before meta")

    # Public source IDs must not look like filesystem paths
    for e in events:
        if e.get("event") != "sources":
            continue
        data = e.get("data") or {}
        if not isinstance(data, dict):
            continue
        for item in data.get("items") or []:
            if not isinstance(item, dict):
                continue
            src = item.get("source")
            if src is None:
                continue
            s = str(src)
            if "/" in s or "\\" in s or s.startswith("data:"):
                raise ValueError(f"sources expose path-like id: {s!r}")


_PUBLIC_FAQ_SOURCE = re.compile(r"^faq-(en|ko|ja|zh)-.+$")


def is_safe_public_source_id(source: str | None) -> bool:
    """True when source is empty, a public FAQ id, or a path-free stem."""
    if source is None or source == "":
        return True
    s = str(source)
    if "/" in s or "\\" in s or "data/onlybook" in s:
        return False
    return True
