"""SSE chat endpoint contract tests (docs/sse-contract.md)."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

# Ensure mock LLM before app import side effects in workers
os.environ.setdefault("MOCK_LLM", "1")

from app.api.schemas import ChatRequest
from app.api.sse_util import (
    is_safe_public_source_id,
    parse_sse_events,
    validate_smoke_sse_body,
)
from app.config import get_settings
from app.main import app

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.setenv("CORS_ORIGINS", "*")
    get_settings.cache_clear()


def _fake_stream_tokens(state: dict[str, Any], *, llm: Any | None = None):
    """Deterministic SSE event sequence matching the frozen contract."""
    session_hint = state.get("session_id")
    yield {
        "event": "meta",
        "data": {
            "language": state.get("language") or "en",
            "route": "simple_retrieve",
            "session_id": session_hint,
        },
    }
    yield {
        "event": "sources",
        "data": {
            "items": [
                {
                    "title": "What is the refund policy?",
                    "score": 0.91,
                    "source": "faq-en-Q18",
                }
            ]
        },
    }
    yield {"event": "token", "data": {"text": "Refunds "}}
    yield {"event": "token", "data": {"text": "within 14 days."}}
    yield {
        "event": "done",
        "data": {
            "finish_reason": "stop",
            "answer": "Refunds within 14 days.",
        },
    }


async def _stream_body(client: AsyncClient, payload: dict[str, Any]) -> tuple[int, str]:
    async with client.stream(
        "POST",
        "/v1/chat/stream",
        json=payload,
    ) as resp:
        body = ""
        async for chunk in resp.aiter_text():
            body += chunk
        return resp.status_code, body


async def test_chat_stream_contract_order_session_and_sources() -> None:
    transport = ASGITransport(app=app)
    req_session = "sse-contract-session-001"
    with patch("app.api.routes_chat.stream_tokens", side_effect=_fake_stream_tokens):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            status, body = await _stream_body(
                client,
                {
                    "message": "What is the refund policy?",
                    "language": "en",
                    "session_id": req_session,
                },
            )
    assert status == 200
    events = parse_sse_events(body)
    names = [e["event"] for e in events]
    assert names[0] == "meta"
    assert names[-1] == "done"
    assert "error" not in names
    assert "meta" in names and "sources" in names and "token" in names

    meta = next(e for e in events if e["event"] == "meta")
    done = next(e for e in events if e["event"] == "done")
    assert isinstance(meta["data"], dict)
    assert isinstance(done["data"], dict)
    assert meta["data"].get("session_id") == req_session
    assert done["data"].get("session_id") == req_session

    sources = next(e for e in events if e["event"] == "sources")
    items = (sources["data"] or {}).get("items") or []
    assert items
    for item in items:
        src = item.get("source")
        assert is_safe_public_source_id(src)
        assert "/" not in str(src)
        assert "data/" not in str(src)
        assert "\\" not in str(src)

    validate_smoke_sse_body(body)


async def test_chat_stream_generates_session_id_when_omitted() -> None:
    transport = ASGITransport(app=app)
    with patch("app.api.routes_chat.stream_tokens", side_effect=_fake_stream_tokens):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            status, body = await _stream_body(
                client, {"message": "Hello support", "language": "en"}
            )
    assert status == 200
    events = parse_sse_events(body)
    meta = next(e for e in events if e["event"] == "meta")
    done = next(e for e in events if e["event"] == "done")
    sid = meta["data"].get("session_id")
    assert sid and isinstance(sid, str)
    assert done["data"].get("session_id") == sid


async def test_malformed_empty_message_rejected() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/chat/stream", json={"message": ""})
    assert resp.status_code == 422


async def test_unsupported_metadata_filter_key_rejected() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            message="hello",
            metadata_filters={"not_a_real_field": "x"},
        )
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/stream",
            json={
                "message": "hello",
                "metadata_filters": {"__proto__": "x"},
            },
        )
    assert resp.status_code == 422


async def test_metadata_filters_propagated_to_stream_state() -> None:
    captured: dict[str, Any] = {}

    def capture_stream(state: dict[str, Any], *, llm: Any | None = None):
        captured.update(state)
        yield from _fake_stream_tokens(state, llm=llm)

    transport = ASGITransport(app=app)
    with patch("app.api.routes_chat.stream_tokens", side_effect=capture_stream):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            status, _body = await _stream_body(
                client,
                {
                    "message": "refund?",
                    "language": "en",
                    "metadata_filters": {"category": "payments", "faq_id": "Q18"},
                },
            )
    assert status == 200
    assert captured.get("metadata_filters") == {
        "category": "payments",
        "faq_id": "Q18",
    }


def test_validate_smoke_sse_body_rejects_token_only() -> None:
    body = "event: token\ndata: {\"text\":\"hi\"}\n\n"
    with pytest.raises(ValueError, match="meta"):
        validate_smoke_sse_body(body)


def test_validate_smoke_sse_body_rejects_error_event() -> None:
    body = (
        "event: meta\ndata: {\"language\":\"en\",\"route\":\"simple_retrieve\"}\n\n"
        "event: sources\ndata: {\"items\":[]}\n\n"
        "event: error\ndata: {\"message\":\"boom\"}\n\n"
        "event: done\ndata: {\"finish_reason\":\"error\"}\n\n"
    )
    with pytest.raises(ValueError, match="error"):
        validate_smoke_sse_body(body)


def test_validate_smoke_sse_body_accepts_valid_stream() -> None:
    body = (
        "event: meta\ndata: "
        + json.dumps({"language": "en", "route": "simple_retrieve", "session_id": "s1"})
        + "\n\n"
        "event: sources\ndata: "
        + json.dumps(
            {
                "items": [
                    {"title": "Refund", "score": 0.9, "source": "faq-en-Q18"}
                ]
            }
        )
        + "\n\n"
        "event: token\ndata: {\"text\":\"ok\"}\n\n"
        "event: done\ndata: "
        + json.dumps({"finish_reason": "stop", "session_id": "s1", "answer": "ok"})
        + "\n\n"
    )
    validate_smoke_sse_body(body)
