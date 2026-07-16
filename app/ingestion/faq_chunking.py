"""
Atomic FAQ chunking: question and answer always co-located.

Oversized answers are split into subchunks that each repeat the question.
Never produce question-only or answer-only FAQ chunks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from app.ingestion.chunking import DocumentChunk
from app.ingestion.faq_parser import FaqEntry, FaqParseResult, parse_faq_document
from app.utils.display_title import resolve_display_title

logger = logging.getLogger(__name__)

DEFAULT_FAQ_MAX_CHARS = 1800
_CHUNK_SCHEMA_VERSION = "faq_atomic_v1"


def format_faq_text(question: str, answer: str) -> str:
    """Canonical retrieval text for one FAQ unit (or subchunk)."""
    q = question.strip()
    a = answer.strip()
    return f"## Question\n{q}\n\n## Answer\n{a}"


def _split_answer_parts(answer: str, *, max_chars: int) -> list[str]:
    """Split long answers on paragraph/sentence boundaries without emptying parts."""
    answer = answer.strip()
    if len(answer) <= max_chars:
        return [answer]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", answer) if p.strip()]
    if not paragraphs:
        paragraphs = [answer]

    parts: list[str] = []
    buf = ""
    for para in paragraphs:
        # Hard-split very long paragraphs
        pieces = [para]
        if len(para) > max_chars:
            pieces = []
            start = 0
            while start < len(para):
                pieces.append(para[start : start + max_chars])
                start += max_chars
        for piece in pieces:
            if not buf:
                buf = piece
            elif len(buf) + 2 + len(piece) <= max_chars:
                buf = f"{buf}\n\n{piece}"
            else:
                parts.append(buf)
                buf = piece
    if buf:
        parts.append(buf)
    # Safety: never return empty
    return [p.strip() for p in parts if p.strip()] or [answer[:max_chars].strip()]


def _chunk_id(source: str, faq_id: str, part_index: int) -> str:
    digest = hashlib.sha1(
        f"{source}:{faq_id}:{part_index}".encode()
    ).hexdigest()[:12]
    return f"faq_{digest}"


def entries_to_chunks(
    entries: list[FaqEntry],
    *,
    source: str,
    title: str | None = None,
    max_chars: int = DEFAULT_FAQ_MAX_CHARS,
) -> list[DocumentChunk]:
    """Convert complete FAQ entries into atomic DocumentChunks."""
    chunks: list[DocumentChunk] = []
    for entry in entries:
        if not entry.is_complete():
            continue
        answer_parts = _split_answer_parts(entry.answer, max_chars=max_chars)
        part_count = len(answer_parts)
        for part_i, ans_part in enumerate(answer_parts):
            # Always include full question with every answer part
            text = format_faq_text(entry.question, ans_part)
            fm = {}
            if isinstance(entry.raw_meta, dict):
                fm = entry.raw_meta.get("front_matter") or {}
            display_title = resolve_display_title(
                {
                    "question": entry.question,
                    "title": title,
                    "category": entry.category,
                    "source": source,
                    "front_matter_title": (fm.get("title") if isinstance(fm, dict) else None),
                },
                text=text,
            )
            # Leave headroom for question when checking size; already enforced on answer
            meta: dict[str, Any] = {
                "doc_type": "faq",
                "chunk_schema_version": _CHUNK_SCHEMA_VERSION,
                "splitter": "faq_atomic",
                "language": entry.language or "",
                "source": source,
                "title": display_title,
                "category": entry.category or "",
                "intent": entry.intent or "",
                "faq_id": entry.faq_id or "",
                "question": entry.question,
                "answer": ans_part,
                "answer_full_length": len(entry.answer),
                "faq_part_index": part_i,
                "faq_part_count": part_count,
                "chunk_index": len(chunks),
            }
            chunks.append(
                DocumentChunk(
                    id=_chunk_id(source, entry.faq_id or str(entry.index), part_i),
                    text=text,
                    # FAQ: raw Q/A is the retrieval text; no LLM contextualization by default
                    contextual_text=text,
                    metadata=meta,
                )
            )
    return chunks


def chunk_faq_document(
    text: str,
    *,
    source: str = "unknown",
    language: str | None = None,
    title: str | None = None,
    explicit_meta: dict[str, Any] | None = None,
    max_chars: int = DEFAULT_FAQ_MAX_CHARS,
) -> tuple[list[DocumentChunk], FaqParseResult]:
    """
    Parse FAQ text and build atomic chunks.

    Returns (chunks, parse_result) so callers can log skips/warnings.
    """
    result = parse_faq_document(
        text,
        source=source,
        language_hint=language,
        explicit_meta=explicit_meta,
    )
    for w in result.warnings:
        logger.warning("%s", w)
    chunks = entries_to_chunks(
        result.entries,
        source=source,
        title=title,
        max_chars=max_chars,
    )
    return chunks, result


def assert_atomic_faq_chunk(chunk: DocumentChunk) -> None:
    """Raise AssertionError if chunk is not a valid atomic FAQ unit."""
    text = chunk.text or ""
    q = str((chunk.metadata or {}).get("question") or "").strip()
    a = str((chunk.metadata or {}).get("answer") or "").strip()
    if not q or not a:
        raise AssertionError("FAQ chunk missing question or answer metadata")
    if q not in text:
        raise AssertionError("FAQ chunk text missing question")
    # Answer body after ## Answer must be non-empty and reflect metadata answer
    if "## Question" not in text or "## Answer" not in text:
        raise AssertionError("FAQ chunk must contain ## Question and ## Answer sections")
    answer_body = text.split("## Answer", 1)[-1].strip()
    if not answer_body:
        raise AssertionError("FAQ chunk has empty answer section")
    # Metadata answer should match body (allow whitespace normalization)
    if a not in text and answer_body not in a and a not in answer_body:
        raise AssertionError("FAQ chunk text missing answer part")
