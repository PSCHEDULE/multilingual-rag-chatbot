"""Unit tests for FAQ vs generic routing in the ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import get_settings


@pytest.fixture(autouse=True)
def _mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOCK_LLM", "1")
    get_settings.cache_clear()


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_pipeline_front_matter_doc_type_overrides_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    When front-matter doc_type conflicts with CLI --doc-type, front-matter wins.

    File body is generic prose but FM says doc_type: faq — even with CLI generic,
    routing must attempt FAQ mode (zero FAQ chunks + warning is acceptable if no Q/A).
    File with FAQ structure + FM doc_type: generic must use generic chunker even if CLI=faq.
    """
    captured: list[Any] = []

    def fake_upsert(chunks, **kwargs):  # type: ignore[no-untyped-def]
        captured.extend(chunks)
        return len(chunks)

    monkeypatch.setattr("app.ingestion.pipeline.upsert_chunks", fake_upsert)

    # FM generic + FAQ-looking body + CLI faq → still generic (FM wins)
    faq_looking = """---
doc_type: generic
---
## Question
Should this be FAQ?

## Answer
Only if routing allows it.
"""
    p1 = _write(tmp_path / "forced_generic.md", faq_looking)

    from app.ingestion.pipeline import run_ingest

    stats = run_ingest(
        p1,
        collection="unused_unit_test",
        recreate=False,
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,
        doc_type="faq",  # CLI wants FAQ
    )
    assert stats["generic_files"] == 1
    assert stats["faq_files"] == 0
    assert all(c.metadata.get("splitter") != "faq_atomic" for c in captured)
    assert all(c.metadata.get("doc_type") != "faq" for c in captured) or all(
        c.metadata.get("chunk_schema_version") != "faq_atomic_v1" for c in captured
    )

    captured.clear()

    # FM faq + CLI generic on a real FAQ body → FAQ atomic
    real_faq = """---
doc_type: faq
faq_id: fm_over_cli
intent: account
---
## Question
How do I log in?

## Answer
Use the Sign In button.
"""
    p2 = _write(tmp_path / "forced_faq.md", real_faq)
    stats2 = run_ingest(
        p2,
        collection="unused_unit_test",
        recreate=False,
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,
        doc_type="generic",  # CLI wants generic
    )
    assert stats2["faq_files"] == 1
    assert stats2["generic_files"] == 0
    assert len(captured) == 1
    assert captured[0].metadata["doc_type"] == "faq"
    assert captured[0].metadata["splitter"] == "faq_atomic"
    assert captured[0].metadata["faq_id"] == "fm_over_cli"
    assert captured[0].metadata["intent"] == "account"


def test_pipeline_faq_contextualize_off_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAQ path must not call contextualize_chunks unless contextualize_faq=True."""
    calls: list[str] = []

    def spy_contextualize(chunks, **kwargs):  # type: ignore[no-untyped-def]
        calls.append("called")
        return chunks

    monkeypatch.setattr("app.ingestion.pipeline.contextualize_chunks", spy_contextualize)
    monkeypatch.setattr(
        "app.ingestion.pipeline.upsert_chunks",
        lambda chunks, **kwargs: len(chunks),
    )

    faq = """## Question
Short Q?

## Answer
Short A.
"""
    _write(tmp_path / "short.md", faq)

    from app.ingestion.pipeline import run_ingest

    run_ingest(
        tmp_path / "short.md",
        collection="unused",
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=False,  # default
        doc_type="faq",
    )
    assert calls == []

    run_ingest(
        tmp_path / "short.md",
        collection="unused",
        prefer_bge=False,
        contextualize=False,
        contextualize_faq=True,
        doc_type="faq",
    )
    assert calls == ["called"]


def test_pipeline_generic_never_uses_faq_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[Any] = []
    monkeypatch.setattr(
        "app.ingestion.pipeline.upsert_chunks",
        lambda chunks, **kwargs: (captured.extend(chunks) or len(chunks)),
    )

    prose = """# Refund Policy

Customers may request a refund within 14 days of purchase.

Approved refunds return to the original payment method within 5–10 business days.
"""
    _write(tmp_path / "policy.md", prose)

    from app.ingestion.pipeline import run_ingest

    stats = run_ingest(
        tmp_path / "policy.md",
        collection="unused",
        prefer_bge=False,
        contextualize=False,
        doc_type="auto",
    )
    assert stats["generic_files"] == 1
    assert stats["faq_files"] == 0
    for c in captured:
        assert c.metadata.get("splitter") != "faq_atomic"
        assert c.metadata.get("chunk_schema_version") != "faq_atomic_v1"
        assert c.metadata.get("doc_type") != "faq"
