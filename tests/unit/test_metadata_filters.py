"""Allowlisted metadata_filters validation and retrieval propagation."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.api.schemas import ChatRequest
from app.config import get_settings
from app.retrieval.metadata_filters import (
    ALLOWED_METADATA_FILTER_KEYS,
    merge_language_and_metadata_filters,
    normalize_metadata_filters,
)


def test_allowlist_includes_payload_fields() -> None:
    assert {"language", "category", "source", "doc_type", "faq_id", "intent"} <= set(
        ALLOWED_METADATA_FILTER_KEYS
    )


def test_normalize_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        normalize_metadata_filters({"unknown": "x"})


def test_normalize_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_metadata_filters({"category": "  "})


def test_chat_request_accepts_allowlisted_filters() -> None:
    req = ChatRequest(
        message="refund?",
        metadata_filters={"category": "payments", "faq_id": "Q18"},
    )
    assert req.metadata_filters == {"category": "payments", "faq_id": "Q18"}


def test_chat_request_rejects_unknown_filter() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="x", metadata_filters={"score_gt": "0.5"})


def test_merge_explicit_language_overrides_auto() -> None:
    merged = merge_language_and_metadata_filters(
        auto_language="ko",
        metadata_filters={"language": "en", "category": "payments"},
    )
    assert merged == {"language": "en", "category": "payments"}


def test_simple_retrieve_passes_metadata_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PREFER_BGE", "false")
    monkeypatch.setenv("QDRANT_COLLECTION", "onlybook_faq")
    monkeypatch.setenv("RETRIEVAL_LANGUAGE_FILTER", "true")
    get_settings.cache_clear()

    captured: dict[str, Any] = {}

    class FakeHit:
        text = "hit"
        score = 1.0
        metadata = {
            "title": "t",
            "source": "s",
            "language": "en",
            "category": "payments",
        }

    class FakeResult:
        hits = [FakeHit()]

    def fake_retrieve(query: str, **kwargs: Any) -> FakeResult:
        captured["query"] = query
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr("app.graph.nodes.retrieve_and_rerank", fake_retrieve)

    from app.graph.nodes import simple_retrieve

    simple_retrieve(
        {
            "messages": [{"role": "user", "content": "refund?"}],
            "language": "en",
            "metadata_filters": {"category": "payments"},
        }
    )
    assert captured.get("metadata_filters") == {
        "language": "en",
        "category": "payments",
    }


def test_hybrid_search_builds_allowlisted_conditions_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """hybrid_search must only emit FieldCondition for allowlisted keys."""
    from app.retrieval import qdrant_store

    monkeypatch.setenv("PREFER_BGE", "false")
    get_settings.cache_clear()

    captured_filters: list[Any] = []

    class FakePoint:
        id = "1"
        score = 0.5
        payload = {"text": "t", "language": "en", "category": "payments"}

    class FakeClient:
        def query_points(self, **kwargs: Any) -> Any:
            captured_filters.append(kwargs.get("prefetch"))
            return type("R", (), {"points": [FakePoint()]})()

        def search(self, **kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr(qdrant_store, "get_qdrant_client", lambda s=None: FakeClient())
    monkeypatch.setattr(
        qdrant_store,
        "get_dense_embedder",
        lambda **kw: type(
            "E",
            (),
            {"embed_query": lambda self, q: [0.1] * 384},
        )(),
    )

    hits = qdrant_store.hybrid_search(
        "refund",
        metadata_filters={"category": "payments", "language": "en"},
        prefer_bge=False,
        top_n=5,
    )
    assert len(hits) == 1
    # Prefetch filters attached
    assert captured_filters
    prefetch = captured_filters[0]
    assert prefetch is not None
    filt = prefetch[0].filter
    assert filt is not None
    keys = {cond.key for cond in filt.must}
    assert keys == {"category", "language"}
