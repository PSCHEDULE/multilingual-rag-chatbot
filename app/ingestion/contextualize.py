"""Contextual Retrieval: LLM-generated situating context per chunk."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings
from app.ingestion.chunking import DocumentChunk
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

CONTEXT_PROMPT = """You are indexing a customer-support knowledge base.
Given the document title and chunk text, write 1–2 concise sentences of context
that situate this chunk within the document (who/what it is about). Do not
repeat the chunk verbatim. Write in the same language as the chunk.

Title: {title}
Language: {language}
Chunk:
{chunk}

Context:"""


def format_contextualize_prompt(
    *,
    title: str,
    language: str,
    chunk: str,
) -> str:
    return CONTEXT_PROMPT.format(title=title, language=language, chunk=chunk)


def contextualize_chunk(
    chunk: DocumentChunk,
    *,
    llm: Any | None = None,
    settings: Settings | None = None,
    enabled: bool | None = None,
) -> DocumentChunk:
    """
    Attach ``contextual_text`` and return a copy-ready chunk.

    When contextualization is disabled or the LLM fails, ``contextual_text``
    falls back to title-prefixed original text.
    """
    cfg = settings or get_settings()
    if enabled is None:
        enabled = cfg.contextualize

    title = str(chunk.metadata.get("title") or chunk.metadata.get("source") or "Document")
    language = str(chunk.metadata.get("language") or "en")
    fallback = f"{title}: {chunk.text}"

    if not enabled:
        return chunk.model_copy(update={"contextual_text": fallback})

    client = llm or get_llm_client(cfg)
    prompt = format_contextualize_prompt(title=title, language=language, chunk=chunk.text)
    try:
        context = client.complete(
            [
                {
                    "role": "system",
                    "content": "You write short indexing context for retrieval systems.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=128,
        ).strip()
        if not context:
            context = fallback
        # Embedding input = context + chunk (Anthropic-style Contextual Retrieval)
        situated = f"{context}\n\n{chunk.text}"
        return chunk.model_copy(update={"contextual_text": situated})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Contextualize failed for %s: %s", chunk.id, exc)
        return chunk.model_copy(update={"contextual_text": fallback})


def contextualize_chunks(
    chunks: list[DocumentChunk],
    *,
    llm: Any | None = None,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    return [contextualize_chunk(c, llm=llm, settings=settings) for c in chunks]
