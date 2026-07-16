"""Unit tests for atomic FAQ chunk creation."""

from pathlib import Path

from app.ingestion.chunking import chunk_document
from app.ingestion.faq_chunking import (
    assert_atomic_faq_chunk,
    chunk_faq_document,
    entries_to_chunks,
    format_faq_text,
)
from app.ingestion.faq_parser import FaqEntry

MULTI = """# mixed

## Question
Q1 text?

## Answer
A1 answer body.

## Question
Q2 text?

## Answer
A2 answer body.
"""

OVERSIZE_ANSWER = "Paragraph one about policy.\n\n" + ("Long detail. " * 80)


def test_format_contains_question_and_answer_sections() -> None:
    text = format_faq_text("How?", "Like this.")
    assert "## Question" in text and "## Answer" in text
    assert "How?" in text and "Like this." in text


def test_atomic_chunk_contains_both_q_and_a() -> None:
    chunks, result = chunk_faq_document(
        """## Question
What is the refund policy?

## Answer
Refunds within 14 days if unused.
""",
        source="faq.md",
        language="en",
    )
    assert result.entries
    assert len(chunks) >= 1
    for c in chunks:
        assert_atomic_faq_chunk(c)
        assert c.metadata["doc_type"] == "faq"
        assert c.metadata["question"]
        assert c.metadata["answer"]
        assert c.metadata["question"] in c.text
        assert c.metadata["answer"] in c.text


def test_multi_faq_one_chunk_each() -> None:
    chunks, result = chunk_faq_document(MULTI, source="multi.md", language="en")
    assert len(result.entries) == 2
    assert len(chunks) == 2
    assert "Q1" in chunks[0].text and "A1" in chunks[0].text
    assert "Q2" in chunks[1].text and "A2" in chunks[1].text
    assert "Q1" not in chunks[1].text or "Q2" in chunks[1].text


def test_oversized_answer_repeats_question_in_each_part() -> None:
    entry = FaqEntry(
        question="What is the long policy?",
        answer=OVERSIZE_ANSWER,
        faq_id="long_1",
        language="en",
        category="policy",
    )
    chunks = entries_to_chunks([entry], source="long.md", max_chars=200)
    assert len(chunks) >= 2
    for c in chunks:
        assert_atomic_faq_chunk(c)
        assert "What is the long policy?" in c.text
        assert c.metadata["faq_part_count"] == len(chunks)
        # Never answer-only
        assert "## Question" in c.text


def test_never_question_only_or_answer_only() -> None:
    chunks, _ = chunk_faq_document(
        """## Question
Only Q

## Answer
Only A with enough text.
""",
        source="x.md",
    )
    for c in chunks:
        # Must not be just the question without answer section content
        after_a = c.text.split("## Answer", 1)[-1].strip()
        assert after_a
        after_q = c.text.split("## Question", 1)[-1].split("## Answer", 1)[0].strip()
        assert after_q


def test_faq_skips_contextual_text_as_raw() -> None:
    chunks, _ = chunk_faq_document(
        """## Question
Q?

## Answer
A.
""",
        source="y.md",
    )
    assert chunks[0].contextual_text == chunks[0].text


def test_generic_chunker_still_used_for_non_faq() -> None:
    text = """# Refund Policy

Customers may request a refund within 14 days.

To start a refund, open Support Center.
"""
    # Structural detection: not FAQ
    from app.ingestion.faq_parser import resolve_doc_type

    assert resolve_doc_type(text=text) == "generic"
    chunks = chunk_document(text, language="en", source="refund.md", prefer_bge_m3=False)
    assert chunks
    # Generic chunks should not claim faq atomic schema
    assert all(c.metadata.get("doc_type") != "faq" for c in chunks) or all(
        c.metadata.get("splitter") != "faq_atomic" for c in chunks
    )


def test_onlybook_files_atomic_for_four_languages() -> None:
    root = Path("data/onlybook_faq")
    if not root.exists():
        return  # corpus optional in some environments
    samples = {
        "ko": next(root.joinpath("ko").glob("Q1_*.md")),
        "en": next(root.joinpath("en").glob("Q1_*.md")),
        "ja": next(root.joinpath("ja").glob("Q1_*.md")),
        "zh": next(root.joinpath("zh").glob("Q1_*.md")),
    }
    for lang, path in samples.items():
        text = path.read_text(encoding="utf-8")
        chunks, result = chunk_faq_document(
            text, source=str(path), language=lang
        )
        assert result.entries, f"no entries for {path}"
        assert chunks, f"no chunks for {path}"
        for c in chunks:
            assert_atomic_faq_chunk(c)
            assert c.metadata.get("language") == lang
