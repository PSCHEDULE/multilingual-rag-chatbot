"""
Isolated FAQ ingestion → Qdrant payload checks.

Uses an in-memory Qdrant client so the active/production collection is never touched.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings

pytestmark = pytest.mark.integration

SHORT_FAQ = """---
doc_type: faq
faq_id: short_01
intent: billing
category: payments
language: en
---
# Payments

## Question
How do I get a refund?

## Answer
Request a refund within 14 days via Support.
"""

# Answer long enough to force subchunks under max_chars=400 in test ingest path,
# but we control via chunking after pipeline uses DEFAULT_FAQ_MAX_CHARS=1800.
# Build an answer > 1800 chars so default pipeline produces multiple parts.
_LONG_BODY = "Detail sentence about the multi-step policy. " * 60  # ~2700 chars
OVERSIZE_FAQ = f"""---
doc_type: faq
faq_id: long_01
intent: policy
language: en
---
## Question
What is the full multi-step refund policy?

## Answer
{_LONG_BODY}
"""

GENERIC_DOC = """# Office Hours

The support team is available Monday through Friday, 9am–6pm KST.

For urgent outages use the status page.
"""


@pytest.fixture
def memory_qdrant(monkeypatch: pytest.MonkeyPatch):
    """Single in-memory Qdrant client shared for the test; never hits real storage."""
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.get_qdrant_client",
        lambda settings=None: client,
    )
    monkeypatch.setenv("MOCK_LLM", "1")
    get_settings.cache_clear()
    return client


def _scroll_all(client: Any, collection: str) -> list[Any]:
    points, _ = client.scroll(
        collection_name=collection,
        limit=100,
        with_payload=True,
        with_vectors=False,
    )
    return list(points)


def test_faq_payload_isolated_collection(
    tmp_path: Path, memory_qdrant: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Verify:
    - short FAQs persist as one Q+A point
    - every oversized FAQ subchunk has full question + non-empty answer fragment
    - FAQ metadata survives into Qdrant payload
    - FAQ contextualization is not called by default
    - generic documents do not use faq_atomic
    """
    collection = f"m8a_faq_payload_test_{uuid.uuid4().hex[:12]}"

    short_path = tmp_path / "short_faq.md"
    long_path = tmp_path / "long_faq.md"
    generic_path = tmp_path / "generic.md"
    short_path.write_text(SHORT_FAQ, encoding="utf-8")
    long_path.write_text(OVERSIZE_FAQ, encoding="utf-8")
    generic_path.write_text(GENERIC_DOC, encoding="utf-8")

    contextualize_calls: list[str] = []

    def spy_contextualize(chunks, **kwargs):  # type: ignore[no-untyped-def]
        contextualize_calls.append("yes")
        return chunks

    monkeypatch.setattr("app.ingestion.pipeline.contextualize_chunks", spy_contextualize)

    from app.ingestion.pipeline import run_ingest

    # --- short FAQ only ---
    stats_short = run_ingest(
        short_path,
        collection=collection,
        recreate=True,
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,  # default path
        doc_type="auto",
    )
    assert stats_short["faq_files"] == 1
    assert stats_short["faq_chunks"] == 1
    assert stats_short["points"] == 1
    assert contextualize_calls == [], "FAQ contextualization must not run by default"

    points = _scroll_all(memory_qdrant, collection)
    assert len(points) == 1
    pl = points[0].payload
    assert pl is not None
    assert pl.get("doc_type") == "faq"
    assert pl.get("chunk_schema_version") == "faq_atomic_v1"
    assert pl.get("faq_id") == "short_01"
    assert pl.get("intent") == "billing"
    assert pl.get("category") == "payments"
    assert pl.get("language") == "en"
    assert pl.get("question") and "refund" in pl["question"].lower()
    assert pl.get("answer") and "14 days" in pl["answer"]
    assert "## Question" in pl["text"] and "## Answer" in pl["text"]
    assert pl["question"] in pl["text"]
    assert pl["answer"] in pl["text"]
    # one Q+A point for short FAQ
    assert pl.get("faq_part_count") == 1
    assert pl.get("faq_part_index") == 0
    # contextual_text stays raw (no LLM prefix)
    assert pl.get("contextual_text") == pl.get("text")

    # --- oversized FAQ (append to same isolated collection without recreate) ---
    contextualize_calls.clear()
    stats_long = run_ingest(
        long_path,
        collection=collection,
        recreate=False,
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,
        doc_type="auto",
    )
    assert stats_long["faq_files"] == 1
    assert stats_long["faq_chunks"] >= 2, "oversized answer must split into subchunks"
    assert contextualize_calls == []

    points = _scroll_all(memory_qdrant, collection)
    long_points = [
        p
        for p in points
        if p.payload and p.payload.get("faq_id") == "long_01"
    ]
    assert len(long_points) >= 2
    full_q = "What is the full multi-step refund policy?"
    for p in long_points:
        pl = p.payload
        assert pl.get("doc_type") == "faq"
        assert pl.get("chunk_schema_version") == "faq_atomic_v1"
        assert pl.get("intent") == "policy"
        assert pl.get("question") == full_q
        assert full_q in pl["text"]
        assert "## Question" in pl["text"] and "## Answer" in pl["text"]
        ans = (pl.get("answer") or "").strip()
        assert ans, "every subchunk must have non-empty answer fragment"
        answer_body = pl["text"].split("## Answer", 1)[-1].strip()
        assert answer_body
        assert pl.get("faq_part_count") == len(long_points)
        assert pl.get("faq_part_index") is not None

    # --- generic document must not use faq_atomic ---
    contextualize_calls.clear()
    stats_gen = run_ingest(
        generic_path,
        collection=collection,
        recreate=False,
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,
        doc_type="auto",
    )
    assert stats_gen["generic_files"] == 1
    assert stats_gen["faq_files"] == 0
    # contextualize=False so spy still not called
    assert contextualize_calls == []

    points = _scroll_all(memory_qdrant, collection)
    generic_points = [
        p
        for p in points
        if p.payload and p.payload.get("source") and "generic.md" in str(p.payload["source"])
    ]
    assert generic_points, "generic doc must produce at least one point"
    for p in generic_points:
        pl = p.payload
        assert pl.get("doc_type") != "faq"
        assert pl.get("chunk_schema_version") != "faq_atomic_v1"
        assert pl.get("faq_id") in (None, "")
        # No FAQ atomic markers required; splitter should not be faq_atomic
        # (may be absent on payload if only selected fields stored — check text/meta)
        assert "faq_atomic" not in str(pl.get("chunk_schema_version") or "")

    # Active collection name must never appear as this test's target
    settings = get_settings()
    assert collection != settings.qdrant_collection
    # In-memory only: real active collection untouched (not even listed here)
    names = {c.name for c in memory_qdrant.get_collections().collections}
    assert collection in names
    assert settings.qdrant_collection not in names or collection != settings.qdrant_collection
