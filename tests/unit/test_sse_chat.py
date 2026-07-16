"""SSE chat endpoint unit tests."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

# Ensure mock LLM before app import side effects in workers
os.environ.setdefault("MOCK_LLM", "1")

from app.config import get_settings
from app.main import app

get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _mock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    get_settings.cache_clear()


async def test_chat_stream_emits_events() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "Hello support", "language": "en"},
        ) as resp:
            assert resp.status_code == 200
            body = ""
            async for chunk in resp.aiter_text():
                body += chunk
    assert "event:" in body or "data:" in body
    assert "done" in body or "token" in body or "meta" in body
