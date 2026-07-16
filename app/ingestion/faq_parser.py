"""
FAQ document detection and parsing.

Supports multiple FAQ Q/A pairs in a single document.
Does not apply to general (non-FAQ) documents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Strict structural markers (markdown headings or labeled lines)
_Q_HEADING = re.compile(
    r"^(#{1,6}\s*)?(question|q)\s*[:：]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_A_HEADING = re.compile(
    r"^(#{1,6}\s*)?(answer|a)\s*[:：]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Split document into FAQ blocks starting at Question headings
_QA_BLOCK = re.compile(
    r"(?:^|\n)(#{1,6}\s*)?(?:question|q)\s*[:：]?\s*\n"
    r"(?P<question>.*?)"
    r"\n(#{1,6}\s*)?(?:answer|a)\s*[:：]?\s*\n"
    r"(?P<answer>.*?)(?=\n(?:#{1,6}\s*)?(?:question|q)\s*[:：]?\s*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_FM_LINE = re.compile(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", re.MULTILINE)


@dataclass
class FaqEntry:
    """One FAQ Q/A pair after parsing."""

    question: str
    answer: str
    faq_id: str | None = None
    category: str | None = None
    intent: str | None = None
    language: str | None = None
    index: int = 0
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return bool(self.question.strip() and self.answer.strip())


@dataclass
class FaqParseResult:
    entries: list[FaqEntry]
    skipped_malformed: int = 0
    warnings: list[str] = field(default_factory=list)
    front_matter: dict[str, str] = field(default_factory=dict)
    detection: str = "none"


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Return (front_matter_dict, body_without_front_matter)."""
    m = _FRONT_MATTER.match(text)
    if not m:
        return {}, text
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        lm = _FM_LINE.match(line.strip())
        if lm:
            fm[lm.group(1).strip().lower()] = lm.group(2).strip().strip("\"'")
    body = text[m.end() :]
    return fm, body


def looks_like_faq(text: str) -> bool:
    """
    Strict structural detection: at least one Question heading and one Answer
    heading in order (Question before Answer somewhere).
    """
    body = parse_front_matter(text)[1]
    q_positions = [m.start() for m in _Q_HEADING.finditer(body)]
    a_positions = [m.start() for m in _A_HEADING.finditer(body)]
    if not q_positions or not a_positions:
        return False
    # At least one Q that is followed later by an A
    for qpos in q_positions:
        if any(apos > qpos for apos in a_positions):
            return True
    return False


def resolve_doc_type(
    *,
    explicit_doc_type: str | None = None,
    cli_doc_type: str | None = None,
    text: str = "",
) -> Literal["faq", "generic"]:
    """
    Detection precedence:
    1. explicit doc_type=faq (metadata / front-matter / function arg)
    2. CLI option
    3. strict Question/Answer structural detection
    4. generic fallback
    """
    for candidate in (explicit_doc_type, cli_doc_type):
        if not candidate:
            continue
        c = candidate.strip().lower()
        if c in {"faq", "generic"}:
            return c  # type: ignore[return-value]
        if c == "auto":
            break
    if looks_like_faq(text):
        return "faq"
    return "generic"


def parse_faq_document(
    text: str,
    *,
    source: str = "unknown",
    language_hint: str | None = None,
    explicit_meta: dict[str, Any] | None = None,
) -> FaqParseResult:
    """
    Parse zero or more FAQ entries from ``text``.

    Malformed entries (missing question or answer) are skipped and counted.
    """
    explicit_meta = explicit_meta or {}
    front_matter, body = parse_front_matter(text)
    warnings: list[str] = []
    skipped = 0
    entries: list[FaqEntry] = []

    matches = list(_QA_BLOCK.finditer(body))
    if not matches:
        # Single Q/A without clean regex (looser fallback within FAQ mode only)
        if looks_like_faq(body):
            warnings.append(f"{source}: FAQ structure detected but no Q/A pairs parsed")
        return FaqParseResult(
            entries=[],
            skipped_malformed=0,
            warnings=warnings,
            front_matter=front_matter,
            detection="structural" if looks_like_faq(body) else "none",
        )

    h1 = None
    hm = _H1.search(body)
    if hm:
        h1 = hm.group(1).strip()

    for i, m in enumerate(matches):
        question = (m.group("question") or "").strip()
        answer = (m.group("answer") or "").strip()
        # If regex over-captured into the next FAQ heading, cut it off
        cut = re.search(
            r"\n#{1,6}\s*(?:question|q)\s*[:：]?\s*$",
            answer,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if cut:
            answer = answer[: cut.start()].strip()
        # Answer must not be (or start as) another Question block
        if re.match(r"#{1,6}\s*(?:question|q)\b", answer, flags=re.IGNORECASE):
            answer = ""
        if not question or not answer:
            skipped += 1
            warnings.append(
                f"{source}: skipped malformed FAQ entry index={i} "
                f"(q_empty={not bool(question)} a_empty={not bool(answer)})"
            )
            continue

        faq_id = _resolve_faq_id(
            explicit_meta=explicit_meta,
            front_matter=front_matter,
            source=source,
            index=i,
        )
        category = _resolve_category(
            explicit_meta=explicit_meta,
            front_matter=front_matter,
            h1=h1,
            source=source,
        )
        intent = _resolve_intent(
            explicit_meta=explicit_meta,
            front_matter=front_matter,
        )
        language = _resolve_language(
            explicit_meta=explicit_meta,
            front_matter=front_matter,
            language_hint=language_hint,
            question=question,
        )

        entries.append(
            FaqEntry(
                question=question,
                answer=answer,
                faq_id=faq_id,
                category=category,
                intent=intent,
                language=language,
                index=i,
                raw_meta={
                    "front_matter": front_matter,
                    "h1": h1,
                },
            )
        )

    return FaqParseResult(
        entries=entries,
        skipped_malformed=skipped,
        warnings=warnings,
        front_matter=front_matter,
        detection="structural",
    )


def _resolve_faq_id(
    *,
    explicit_meta: dict[str, Any],
    front_matter: dict[str, str],
    source: str,
    index: int,
) -> str:
    """
    Precedence: explicit meta > front_matter > filename stem patterns > source+index.
    """
    for key in ("faq_id", "id", "source_qid"):
        if explicit_meta.get(key):
            return str(explicit_meta[key])
    for key in ("faq_id", "id", "source_qid"):
        if front_matter.get(key):
            return front_matter[key]
    stem = Path(source).stem
    # e.g. Q18_faq_18_ko → prefer Q18 or faq_18_ko
    m = re.match(r"^(Q\d+)", stem, re.IGNORECASE)
    if m:
        base = m.group(1)
        return base if index == 0 else f"{base}_{index}"
    m2 = re.search(r"(faq_\d+[_\w]*)", stem, re.IGNORECASE)
    if m2:
        base = m2.group(1)
        return base if index == 0 else f"{base}_{index}"
    return f"{stem}_{index}" if index else stem


def _resolve_category(
    *,
    explicit_meta: dict[str, Any],
    front_matter: dict[str, str],
    h1: str | None,
    source: str,
) -> str:
    """Precedence: explicit meta > front_matter > H1 > path stem."""
    if explicit_meta.get("category"):
        return str(explicit_meta["category"])
    if front_matter.get("category"):
        return front_matter["category"]
    if h1:
        return h1
    return Path(source).stem


def _resolve_intent(
    *,
    explicit_meta: dict[str, Any],
    front_matter: dict[str, str],
) -> str:
    """Precedence: explicit meta > front_matter > empty string."""
    if explicit_meta.get("intent"):
        return str(explicit_meta["intent"])
    if front_matter.get("intent"):
        return front_matter["intent"]
    return ""


def _resolve_language(
    *,
    explicit_meta: dict[str, Any],
    front_matter: dict[str, str],
    language_hint: str | None,
    question: str,
) -> str | None:
    """
    Precedence: explicit meta > front_matter > language_hint (e.g. parent dir)
    > detect from question (caller may refine).
    """
    from app.utils.language import detect_language, normalize_language

    for raw in (
        explicit_meta.get("language"),
        explicit_meta.get("lang"),
        front_matter.get("language"),
        front_matter.get("lang"),
        language_hint,
    ):
        if raw:
            n = normalize_language(str(raw))
            if n:
                return n
    return detect_language(question)
