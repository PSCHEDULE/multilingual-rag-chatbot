"""Integration hybrid + language filter (skip without Qdrant)."""

import pytest

from app.config import get_settings
from app.retrieval.hybrid import retrieve_and_rerank
from app.retrieval.qdrant_store import qdrant_reachable

pytestmark = pytest.mark.integration


def test_hybrid_language_filter() -> None:
    if not qdrant_reachable():
        pytest.skip("Qdrant not reachable")

    settings = get_settings()
    # Assumes prior ingest into default collection; if empty, skip
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url)
    try:
        info = client.get_collection(settings.qdrant_collection)
    except Exception:
        pytest.skip("collection missing — run ingest first")
    if not info.points_count:
        pytest.skip("empty collection")

    result = retrieve_and_rerank(
        "환불 정책이 어떻게 되나요?",
        language="ko",
        top_n=20,
        top_k=5,
        prefer_bge=False,
    )
    assert 1 <= len(result) <= 5
    assert all(hasattr(h, "score") for h in result.hits)
    assert all(h.metadata.get("language") == "ko" for h in result.hits)
    assert result.metrics.retrieval_ms >= 0
    assert result.metrics.rerank_ms >= 0
