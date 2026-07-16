"""Display title resolution for FAQ sources."""

from app.ingestion.faq_chunking import entries_to_chunks, format_faq_text
from app.ingestion.faq_parser import FaqEntry
from app.utils.display_title import looks_like_internal_title, resolve_display_title


def test_looks_like_internal_title() -> None:
    assert looks_like_internal_title("Q15 Faq 15 Ja")
    assert looks_like_internal_title("Q18 Faq 18 Ko")
    assert looks_like_internal_title("Q1 Faq 01 En")
    assert not looks_like_internal_title("What is the refund policy?")
    assert not looks_like_internal_title("환불 정책은 어떻게 되나요?")


def test_prefer_question_over_filename_title() -> None:
    title = resolve_display_title(
        {
            "title": "Q15 Faq 15 Ja",
            "question": "無料体験が終了すると自動的に課金されますか？",
            "source": "data/onlybook_faq/ja/Q15_faq_15_ja.md",
        }
    )
    assert title == "無料体験が終了すると自動的に課金されますか？"


def test_extract_question_from_chunk_text() -> None:
    text = format_faq_text("How do I get a refund?", "Within 7 days.")
    title = resolve_display_title(
        {"title": "Q18 Faq 18 En", "source": "Q18_faq_18_en.md"},
        text=text,
    )
    assert title == "How do I get a refund?"


def test_fallback_category_when_no_question() -> None:
    title = resolve_display_title(
        {
            "title": "Q9 Faq 09 En",
            "category": "payment_refund",
            "source": "x.md",
        }
    )
    assert "payment" in title.lower() or "Payment" in title


def test_entries_to_chunks_sets_question_as_title() -> None:
    entry = FaqEntry(
        question="Will I be charged automatically after the free trial?",
        answer="Not without a payment method.",
        faq_id="Q15",
        category="plans_subscription",
        language="en",
    )
    chunks = entries_to_chunks(
        [entry],
        source="data/onlybook_faq/en/Q15_faq_15_en.md",
        title="Q15 Faq 15 En",  # pipeline still passes filename title as hint only
    )
    assert len(chunks) == 1
    assert chunks[0].metadata["title"] == entry.question
    assert chunks[0].metadata["question"] == entry.question


def test_serve_time_style_resolution() -> None:
    """Simulate hybrid hit metadata from existing BGE payload (filename title)."""
    meta = {
        "title": "Q18 Faq 18 Ko",
        "question": "환불 정책은 어떻게 되나요?",
        "source": "data/onlybook_faq/ko/Q18_faq_18_ko.md",
        "language": "ko",
    }
    assert resolve_display_title(meta) == "환불 정책은 어떻게 되나요?"
