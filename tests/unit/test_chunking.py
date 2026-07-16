"""Unit tests for semantic / CJK chunking."""

from pathlib import Path

from app.ingestion.chunking import BagOfNgramsEmbedding, chunk_document, chunk_files

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "sample_docs"


def test_chunk_document_non_empty_with_metadata() -> None:
    text = (
        "환불은 14일 이내에 가능합니다. 고객센터에서 요청하세요.\n\n"
        "배송은 3-7일이 걸립니다. 추적 번호가 이메일로 발송됩니다."
    )
    chunks = chunk_document(
        text,
        language="ko",
        source="test/ko.md",
        title="Test",
        prefer_bge_m3=False,
        embed_model=BagOfNgramsEmbedding(),
    )
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
    assert all(c.metadata.get("language") == "ko" for c in chunks)
    assert all(c.metadata.get("source") == "test/ko.md" for c in chunks)


def test_explicit_language_on_english_text() -> None:
    chunks = chunk_document(
        "Refunds are available within 14 days of purchase.",
        language="ja",
        source="x.md",
        prefer_bge_m3=False,
    )
    assert chunks[0].metadata["language"] == "ja"


def test_sample_docs_all_languages_produce_chunks() -> None:
    paths = sorted(str(p) for p in SAMPLE.rglob("*.md"))
    assert len(paths) >= 12
    chunks = chunk_files(paths, prefer_bge_m3=False)
    assert all(c.text.strip() for c in chunks)
    langs = {c.metadata["language"] for c in chunks}
    assert langs >= {"ko", "en", "ja", "zh"}
    # at least one chunk per language
    for lang in ("ko", "en", "ja", "zh"):
        assert any(c.metadata["language"] == lang for c in chunks)
