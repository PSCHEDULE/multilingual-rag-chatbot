"""Client-safe source ids for SSE sources event."""

from app.utils.public_source import infer_language_from_source, public_source_id


def test_language_prefixed_faq_id() -> None:
    assert (
        public_source_id(
            "data/onlybook_faq/ja/Q15_faq_15_ja.md",
            faq_id="Q15",
            language="ja",
        )
        == "faq-ja-Q15"
    )
    assert (
        public_source_id(
            "data/onlybook_faq/ko/Q18_faq_18_ko.md",
            faq_id="Q18",
            language="ko",
        )
        == "faq-ko-Q18"
    )
    assert (
        public_source_id(
            None,
            faq_id="Q18",
            language="en",
        )
        == "faq-en-Q18"
    )


def test_unique_across_languages_same_qid() -> None:
    ids = {
        public_source_id(faq_id="Q15", language=lang)
        for lang in ("en", "ko", "ja", "zh")
    }
    assert ids == {"faq-en-Q15", "faq-ko-Q15", "faq-ja-Q15", "faq-zh-Q15"}
    assert len(ids) == 4


def test_infer_language_from_path_when_language_omitted() -> None:
    assert (
        public_source_id(
            "data/onlybook_faq/zh/Q1_faq_01_zh.md",
            faq_id="Q1",
        )
        == "faq-zh-Q1"
    )
    assert infer_language_from_source("data/onlybook_faq/ja/Q15_faq_15_ja.md") == "ja"


def test_filename_stem_without_faq_id() -> None:
    assert (
        public_source_id("data/onlybook_faq/en/refund_policy.md")
        == "refund_policy"
    )


def test_no_full_path_leak() -> None:
    out = public_source_id(
        "data/onlybook_faq/ko/Q18_faq_18_ko.md",
        faq_id="Q18",
        language="ko",
    )
    assert out == "faq-ko-Q18"
    assert "data/" not in out
    assert "/" not in out
