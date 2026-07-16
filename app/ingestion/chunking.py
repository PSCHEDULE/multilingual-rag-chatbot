"""
CJK-aware semantic chunking.

Primary production path: LlamaIndex ``SemanticSplitterNodeParser`` with BGE-M3
embeddings (when available). Offline / CI path uses the same splitter API with a
lightweight bag-of-n-grams embedder so gates run without multi-GB model downloads.

See ``docs/chunking.md`` for rationale and manual quality review results.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.utils.language import SupportedLanguage, detect_language

logger = logging.getLogger(__name__)

# Sentence-ish boundaries for CJK + Latin (not used as sole English recursive splitter)
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?。！？…])\s+|(?<=\n)\s*(?=\n)|(?<=\n)(?=[#\-])",
)


class DocumentChunk(BaseModel):
    """One retrieval-oriented text unit with metadata."""

    id: str
    text: str
    contextual_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BagOfNgramsEmbedding:
    """
    Lightweight deterministic embedder for offline semantic splitting.

    Production should use BGE-M3 via ``build_embed_model(prefer_bge_m3=True)``.
    """

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> list[float]:
        import math

        vec = [0.0] * self.dim
        t = text.lower().strip()
        if not t:
            return vec
        for i in range(len(t) - self.ngram + 1):
            gram = t[i : i + self.ngram]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def aget_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[list[float]]:
        return [self._vec(t) for t in texts]


def build_embed_model(
    *,
    prefer_bge_m3: bool = True,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> Any:
    """
    Build an embedding model for semantic splitting.

    Tries HuggingFace BGE-M3 when ``prefer_bge_m3`` and deps are available;
    otherwise returns a LlamaIndex-compatible lightweight embedder.
    """
    cfg = settings or get_settings()
    name = model_name or cfg.embedding_model

    if prefer_bge_m3:
        try:
            from llama_index.core.embeddings import resolve_embed_model

            # resolve_embed_model may map local models; try HF path explicitly first
            try:
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding

                logger.info("Loading HuggingFace embedding model %s for chunking", name)
                return HuggingFaceEmbedding(model_name=name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("HuggingFaceEmbedding unavailable: %s", exc)
                try:
                    return resolve_embed_model(f"local:{name}")
                except Exception as exc2:  # noqa: BLE001
                    logger.warning(
                        "BGE-M3/HF embedder unavailable (%s); using offline n-gram embedder",
                        exc2,
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("llama_index embeddings not fully available: %s", exc)

    # Wrap bag-of-ngrams as LlamaIndex BaseEmbedding if possible
    try:
        from llama_index.core.base.embeddings.base import BaseEmbedding

        class _LIBagEmbed(BaseEmbedding):
            dim: int = 256

            def _get_query_embedding(self, query: str) -> list[float]:
                return BagOfNgramsEmbedding(dim=self.dim).get_text_embedding(query)

            def _get_text_embedding(self, text: str) -> list[float]:
                return BagOfNgramsEmbedding(dim=self.dim).get_text_embedding(text)

            def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
                return BagOfNgramsEmbedding(dim=self.dim).get_text_embedding_batch(texts)

            async def _aget_query_embedding(self, query: str) -> list[float]:
                return self._get_query_embedding(query)

            async def _aget_text_embedding(self, text: str) -> list[float]:
                return self._get_text_embedding(text)

        return _LIBagEmbed()
    except Exception:
        return BagOfNgramsEmbedding()


def _sentence_units(text: str) -> list[str]:
    """Split into coarse units preserving CJK punctuation boundaries."""
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    # Prefer blank-line paragraphs first for markdown FAQs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        parts = [s.strip() for s in _SENTENCE_SPLIT.split(para) if s and s.strip()]
        if not parts:
            units.append(para)
        else:
            units.extend(parts)
    return units


def _chunk_ids(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text[:64]}".encode()).hexdigest()[:12]
    return f"chk_{digest}"


def chunk_text_semantic(
    text: str,
    *,
    language: SupportedLanguage | None = None,
    source: str = "unknown",
    title: str | None = None,
    category: str | None = None,
    embed_model: Any | None = None,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int | None = None,
    settings: Settings | None = None,
    prefer_bge_m3: bool = False,
) -> list[DocumentChunk]:
    """
    Chunk text with SemanticSplitterNodeParser (primary strategy).

    Falls back to language-aware paragraph/sentence packing if the splitter
    cannot run or returns no nodes.
    """
    cfg = settings or get_settings()
    lang = language or detect_language(text)
    threshold = (
        breakpoint_percentile_threshold
        if breakpoint_percentile_threshold is not None
        else cfg.chunk_breakpoint_percentile
    )
    model = embed_model or build_embed_model(prefer_bge_m3=prefer_bge_m3, settings=cfg)

    chunks: list[DocumentChunk] = []
    try:
        from llama_index.core import Document
        from llama_index.core.node_parser import SemanticSplitterNodeParser

        parser = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=threshold,
            embed_model=model,
        )
        doc = Document(
            text=text,
            metadata={
                "language": lang,
                "source": source,
                "title": title or source,
                "category": category or "",
            },
        )
        nodes = parser.get_nodes_from_documents([doc])
        for i, node in enumerate(nodes):
            body = (node.get_content() or "").strip()
            if not body:
                continue
            meta = {
                "language": lang,
                "source": source,
                "title": title or source,
                "category": category or "",
                "chunk_index": i,
                "splitter": "semantic_bge_or_compat",
            }
            chunks.append(
                DocumentChunk(
                    id=_chunk_ids(source, i, body),
                    text=body,
                    metadata=meta,
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic splitter failed (%s); using CJK sentence packer", exc)

    if not chunks:
        chunks = _fallback_pack(
            text,
            language=lang,
            source=source,
            title=title,
            category=category,
        )
    return chunks


def _fallback_pack(
    text: str,
    *,
    language: SupportedLanguage,
    source: str,
    title: str | None,
    category: str | None,
    max_chars: int = 600,
) -> list[DocumentChunk]:
    """Pack sentence units without English-only whitespace recursion as sole strategy."""
    units = _sentence_units(text)
    if not units:
        return []
    packed: list[str] = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + 1 + len(u) <= max_chars:
            buf = f"{buf}\n{u}" if "\n" in buf or u.startswith("#") else f"{buf} {u}"
        else:
            packed.append(buf)
            buf = u
    if buf:
        packed.append(buf)

    out: list[DocumentChunk] = []
    for i, body in enumerate(packed):
        body = body.strip()
        if not body:
            continue
        out.append(
            DocumentChunk(
                id=_chunk_ids(source, i, body),
                text=body,
                metadata={
                    "language": language,
                    "source": source,
                    "title": title or source,
                    "category": category or "",
                    "chunk_index": i,
                    "splitter": "cjk_sentence_pack",
                },
            )
        )
    return out


def chunk_document(
    text: str,
    *,
    language: str | None = None,
    source: str = "unknown",
    title: str | None = None,
    category: str | None = None,
    embed_model: Any | None = None,
    prefer_bge_m3: bool = False,
    settings: Settings | None = None,
) -> list[DocumentChunk]:
    """Public entry: detect language and produce non-empty chunks."""
    from app.utils.language import normalize_language

    lang = normalize_language(language) if language else None
    lang = lang or detect_language(text)
    return chunk_text_semantic(
        text,
        language=lang,
        source=source,
        title=title,
        category=category,
        embed_model=embed_model,
        prefer_bge_m3=prefer_bge_m3,
        settings=settings,
    )


def chunk_files(
    paths: Sequence[str],
    *,
    prefer_bge_m3: bool = False,
) -> list[DocumentChunk]:
    """Chunk multiple files; language inferred from parent dir name when possible."""
    from pathlib import Path

    from app.utils.language import normalize_language

    all_chunks: list[DocumentChunk] = []
    embed = build_embed_model(prefer_bge_m3=prefer_bge_m3)
    for p in paths:
        path = Path(p)
        raw = path.read_text(encoding="utf-8")
        parent_lang = normalize_language(path.parent.name)
        title = path.stem.replace("_", " ").title()
        all_chunks.extend(
            chunk_document(
                raw,
                language=parent_lang,
                source=str(path.as_posix()),
                title=title,
                category=path.stem,
                embed_model=embed,
                prefer_bge_m3=prefer_bge_m3,
            )
        )
    return all_chunks


def new_chunk_id() -> str:
    return f"chk_{uuid.uuid4().hex[:12]}"
