"""Unit tests for contextual retrieval prompts (offline)."""

from app.ingestion.chunking import DocumentChunk
from app.ingestion.contextualize import contextualize_chunk, format_contextualize_prompt
from app.llm.client import MockLLMClient


def test_format_prompt_contains_title_and_chunk() -> None:
    p = format_contextualize_prompt(
        title="Refund Policy",
        language="en",
        chunk="Refunds within 14 days.",
    )
    assert "Refund Policy" in p
    assert "Refunds within 14 days" in p


def test_contextualize_with_mock_llm() -> None:
    chunk = DocumentChunk(
        id="c1",
        text="Refunds within 14 days.",
        metadata={"title": "Refund Policy", "language": "en", "source": "en/r.md"},
    )
    out = contextualize_chunk(chunk, llm=MockLLMClient(), enabled=True)
    assert out.contextual_text
    assert "Refunds within 14 days" in out.contextual_text
