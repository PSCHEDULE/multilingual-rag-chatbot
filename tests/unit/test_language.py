"""Unit tests for language detection."""

from app.utils.language import detect_language, normalize_language


def test_explicit_language_wins_over_text() -> None:
    text = "This is clearly English content about refunds."
    assert detect_language(text, explicit="ko") == "ko"
    assert detect_language(text, explicit="ja") == "ja"


def test_normalize_aliases() -> None:
    assert normalize_language("Korean") == "ko"
    assert normalize_language("zh-CN") == "zh"
    assert normalize_language("jp") == "ja"


def test_detect_korean() -> None:
    assert detect_language("환불 정책이 어떻게 되나요? 주문 번호를 알려주세요.") == "ko"


def test_detect_japanese() -> None:
    assert detect_language("返金ポリシーについて教えてください。注文番号が必要です。") == "ja"


def test_detect_chinese() -> None:
    assert detect_language("退款政策是什么？请提供订单号。") == "zh"


def test_detect_english() -> None:
    assert detect_language("How do I request a refund for my order?") == "en"
