"""User-facing source / document title resolution."""

from __future__ import annotations

import re
from typing import Any

# Filename-stem titles like "Q15 Faq 15 Ja" or "Q1 Faq 01 En"
_INTERNAL_FAQ_TITLE = re.compile(
    r"^Q\d+\s+Faq\s+\d+(\s+[A-Za-z]{2})?$",
    re.IGNORECASE,
)
# Bare stems / paths sometimes used as title
_INTERNAL_STEM = re.compile(
    r"^Q\d+[_\s-]faq[_\s-]\d+",
    re.IGNORECASE,
)


def looks_like_internal_title(title: str | None) -> bool:
    """True if title looks like a filename-derived FAQ id, not a user question."""
    if not title or not str(title).strip():
        return True
    t = str(title).strip()
    if _INTERNAL_FAQ_TITLE.match(t):
        return True
    if _INTERNAL_STEM.match(t.replace(" ", "_")):
        return True
    # Entirely path-like
    if "/" in t or t.endswith(".md"):
        return True
    return False


def _question_from_chunk_text(text: str | None) -> str | None:
    """Extract question body from atomic FAQ chunk text if present."""
    if not text:
        return None
    if "## Question" not in text:
        return None
    body = text.split("## Question", 1)[-1]
    if "## Answer" in body:
        body = body.split("## Answer", 1)[0]
    q = body.strip()
    return q or None


def resolve_display_title(
    metadata: dict[str, Any] | None = None,
    text: str | None = None,
    *,
    title: str | None = None,
    question: str | None = None,
    category: str | None = None,
    source: str | None = None,
    front_matter_title: str | None = None,
) -> str:
    """
    Resolve a user-facing title for FAQ / support sources.

    Priority:
      1. FAQ question text (metadata or explicit)
      2. Front-matter / explicit curated title (if not internal-looking)
      3. Question extracted from chunk text (## Question …)
      4. Existing title if it does not look like an internal id
      5. Category (slug humanized lightly)
      6. Source path stem (last resort)
    """
    meta = dict(metadata or {})
    q = (question if question is not None else meta.get("question")) or ""
    q = str(q).strip()
    if q:
        return q

    fm_title = (
        front_matter_title
        if front_matter_title is not None
        else meta.get("front_matter_title") or meta.get("fm_title")
    )
    if fm_title and str(fm_title).strip() and not looks_like_internal_title(str(fm_title)):
        return str(fm_title).strip()

    from_text = _question_from_chunk_text(text if text is not None else meta.get("text"))
    if from_text:
        return from_text

    t = title if title is not None else meta.get("title")
    if t and str(t).strip() and not looks_like_internal_title(str(t)):
        return str(t).strip()

    cat = category if category is not None else meta.get("category")
    if cat and str(cat).strip() and not looks_like_internal_title(str(cat)):
        # light humanize: plans_subscription → Plans subscription
        raw = str(cat).strip().replace("_", " ").replace("-", " ")
        return raw[:1].upper() + raw[1:] if raw else raw

    src = source if source is not None else meta.get("source")
    if src:
        from pathlib import Path

        stem = Path(str(src)).stem.replace("_", " ").strip()
        if stem:
            return stem

    if t and str(t).strip():
        return str(t).strip()
    return "Document"
