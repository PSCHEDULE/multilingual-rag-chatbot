"""Unit tests for FAQ detection and multi-entry parsing."""

from app.ingestion.faq_parser import (
    looks_like_faq,
    parse_faq_document,
    resolve_doc_type,
)

SINGLE_FAQ = """# service_start

## Question
What kind of service is onlybook.ai?

## Answer
onlybook.ai is an AI-powered book creation service.
"""

MULTI_FAQ = """# account

## Question
How do I sign up?

## Answer
Click Sign Up.

## Question
How do I reset my password?

## Answer
Use Forgot Password on the login screen.
"""

MALFORMED = """# broken

## Question
Only a question, no answer section.

## Answer

## Question
Another orphan question without answer.
"""

GENERIC = """# Refund Policy

Customers may request a refund within 14 days of purchase.

Approved refunds return to the original payment method.
"""


def test_looks_like_faq_true_for_qa_markdown() -> None:
    assert looks_like_faq(SINGLE_FAQ) is True
    assert looks_like_faq(MULTI_FAQ) is True


def test_looks_like_faq_false_for_generic_prose() -> None:
    assert looks_like_faq(GENERIC) is False


def test_resolve_doc_type_precedence_explicit() -> None:
    # Explicit wins even if text is generic
    assert resolve_doc_type(explicit_doc_type="faq", text=GENERIC) == "faq"
    assert resolve_doc_type(explicit_doc_type="generic", text=SINGLE_FAQ) == "generic"


def test_resolve_doc_type_cli_over_structure() -> None:
    assert resolve_doc_type(cli_doc_type="generic", text=SINGLE_FAQ) == "generic"
    assert resolve_doc_type(cli_doc_type="faq", text=GENERIC) == "faq"


def test_resolve_doc_type_front_matter_wins_over_cli_conflict() -> None:
    """
    Precedence when CLI --doc-type conflicts with front-matter doc_type:
    1. front-matter / explicit metadata
    2. CLI --doc-type
    3. structural Q/A detection
    4. generic

    Front-matter always wins the conflict (CLI cannot override explicit FM).
    """
    # FM says faq, CLI says generic → faq
    assert (
        resolve_doc_type(
            explicit_doc_type="faq",
            cli_doc_type="generic",
            text=GENERIC,
        )
        == "faq"
    )
    # FM says generic, CLI says faq → generic
    assert (
        resolve_doc_type(
            explicit_doc_type="generic",
            cli_doc_type="faq",
            text=SINGLE_FAQ,
        )
        == "generic"
    )
    # CLI alone still works when FM absent
    assert (
        resolve_doc_type(explicit_doc_type=None, cli_doc_type="faq", text=GENERIC)
        == "faq"
    )


def test_resolve_doc_type_structural_then_generic() -> None:
    assert resolve_doc_type(text=SINGLE_FAQ) == "faq"
    assert resolve_doc_type(text=GENERIC) == "generic"


def test_parse_single_faq() -> None:
    result = parse_faq_document(SINGLE_FAQ, source="en/Q1_faq_01_en.md", language_hint="en")
    assert result.skipped_malformed == 0
    assert len(result.entries) == 1
    e = result.entries[0]
    assert "onlybook.ai" in e.question
    assert "AI-powered" in e.answer
    assert e.language == "en"
    assert e.category == "service_start"
    assert e.faq_id  # deterministic from filename


def test_parse_multiple_faqs_in_one_document() -> None:
    result = parse_faq_document(MULTI_FAQ, source="docs/account.md", language_hint="en")
    assert len(result.entries) == 2
    assert "sign up" in result.entries[0].question.lower()
    assert "password" in result.entries[1].question.lower()
    assert result.entries[0].is_complete()
    assert result.entries[1].is_complete()


def test_malformed_faq_skipped() -> None:
    result = parse_faq_document(MALFORMED, source="broken.md")
    assert len(result.entries) == 0
    assert result.skipped_malformed >= 1 or result.warnings


def test_faq_id_from_filename() -> None:
    result = parse_faq_document(SINGLE_FAQ, source="data/onlybook_faq/ko/Q18_faq_18_ko.md")
    assert result.entries
    assert result.entries[0].faq_id.upper().startswith("Q18")


def test_language_hint_from_parent() -> None:
    result = parse_faq_document(SINGLE_FAQ, source="x.md", language_hint="ja")
    assert result.entries[0].language == "ja"
