"""Language detection and normalization for KO/EN/JA/ZH."""

from __future__ import annotations

import re
from typing import Literal

SupportedLanguage = Literal["ko", "en", "ja", "zh"]

SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"ko", "en", "ja", "zh"})

# langdetect / common aliases → supported codes
_ALIAS: dict[str, SupportedLanguage] = {
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
    "en": "en",
    "eng": "en",
    "english": "en",
    "ja": "ja",
    "jp": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "chinese": "zh",
    "cmn": "zh",
}


def normalize_language(code: str | None) -> SupportedLanguage | None:
    """Map free-form language labels to supported codes, or None if unknown."""
    if not code:
        return None
    key = code.strip().lower().replace("_", "-")
    if key in _ALIAS:
        return _ALIAS[key]
    # zh-XX → zh
    if key.startswith("zh"):
        return "zh"
    return None


def _script_heuristic(text: str) -> SupportedLanguage | None:
    """Fast script-based hint when langdetect is weak on short CJK strings."""
    if not text or not text.strip():
        return None
    sample = text[:2000]
    hangul = len(re.findall(r"[\uac00-\ud7a3]", sample))
    hiragana_katakana = len(re.findall(r"[\u3040-\u30ff]", sample))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", sample))
    latin = len(re.findall(r"[A-Za-z]", sample))

    scores = {
        "ko": hangul,
        "ja": hiragana_katakana + (cjk * 0.3 if hiragana_katakana else 0),
        "zh": cjk if hiragana_katakana == 0 and hangul == 0 else 0,
        "en": latin,
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return None
    # Require a clear winner for short text
    total = sum(scores.values()) or 1
    if scores[best] / total < 0.15 and best != "en":
        return None
    return best  # type: ignore[return-value]


def detect_language(
    text: str,
    *,
    explicit: str | None = None,
    default: SupportedLanguage = "en",
) -> SupportedLanguage:
    """
    Detect query/document language.

    Explicit parameter always wins when it maps to a supported language.
    """
    forced = normalize_language(explicit)
    if forced is not None:
        return forced

    if not text or not text.strip():
        return default

    hint = _script_heuristic(text)
    # Strong Hangul / kana signals are reliable
    if hint in {"ko", "ja"}:
        return hint

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        code = detect(text)
        mapped = normalize_language(code)
        if mapped is not None:
            return mapped
    except Exception:
        pass

    if hint is not None:
        return hint
    return default
