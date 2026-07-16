"""Sanitize source identifiers for client-facing payloads."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_LANG_CODES = frozenset({"ko", "en", "ja", "zh"})
# e.g. Q15_faq_15_ja, Q18_faq_18_ko
_STEM_LANG = re.compile(r"_(ko|en|ja|zh)$", re.IGNORECASE)


def _normalize_lang(raw: str | None) -> str | None:
    if not raw:
        return None
    code = str(raw).strip().lower()
    if code in _LANG_CODES:
        return code
    # BCP-47-ish: ja-JP → ja
    primary = code.split("-", 1)[0]
    if primary in _LANG_CODES:
        return primary
    return None


def infer_language_from_source(source: str | None) -> str | None:
    """Best-effort language from path parent (…/ja/file.md) or filename suffix."""
    if not source:
        return None
    s = str(source).strip().replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    # parent directory name
    if len(parts) >= 2:
        parent = _normalize_lang(parts[-2])
        if parent:
            return parent
    stem = Path(parts[-1]).stem if parts else ""
    m = _STEM_LANG.search(stem)
    if m:
        return m.group(1).lower()
    return None


def public_source_id(
    source: str | None = None,
    *,
    faq_id: str | None = None,
    language: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Client-safe source label for SSE / API (no full internal paths).

    Preference:
      1. Language-unique FAQ id: ``faq-{lang}-{faq_id}`` (e.g. faq-ja-Q15)
      2. Filename stem only (e.g. Q15_faq_15_ja) when faq_id missing
      3. last path segment
      4. empty string
    """
    meta = metadata or {}
    fid = (faq_id if faq_id is not None else meta.get("faq_id")) or ""
    fid = str(fid).strip()
    raw = source if source is not None else meta.get("source")

    lang = _normalize_lang(
        language if language is not None else meta.get("language")
    )
    if not lang:
        lang = infer_language_from_source(str(raw) if raw else None)

    if fid:
        # Normalize faq_id casing for display: keep leading Q + digits as given, strip spaces
        fid_clean = re.sub(r"\s+", "", fid)
        if lang:
            return f"faq-{lang}-{fid_clean}"
        # faq_id without language: still prefix to mark as FAQ (best effort)
        return f"faq-{fid_clean}"

    if not raw:
        return ""
    s = str(raw).strip().replace("\\", "/")
    name = Path(s).name
    if name.endswith(".md") or name.endswith(".txt") or name.endswith(".markdown"):
        return Path(name).stem
    if "/" in s:
        return Path(s).name or s
    return s
