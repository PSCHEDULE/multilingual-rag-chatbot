"""Load → chunk → contextualize → embed → upsert."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from app.config import Settings, get_settings
from app.ingestion.chunking import DocumentChunk, chunk_document
from app.ingestion.contextualize import contextualize_chunks
from app.ingestion.faq_chunking import chunk_faq_document
from app.ingestion.faq_parser import resolve_doc_type
from app.retrieval.qdrant_store import upsert_chunks
from app.utils.language import normalize_language

logger = logging.getLogger(__name__)

DocTypeOpt = Literal["faq", "generic", "auto"] | None


def load_documents(path: str | Path) -> list[tuple[str, str, str | None]]:
    """
    Return list of (text, source, language_hint).

    Language hint comes from parent directory name when it is ko/en/ja/zh.
    """
    root = Path(path)
    files: list[Path] = []
    if root.is_file():
        files = [root]
    else:
        for ext in ("*.md", "*.txt", "*.markdown"):
            files.extend(sorted(root.rglob(ext)))

    docs: list[tuple[str, str, str | None]] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        lang = normalize_language(f.parent.name)
        docs.append((text, f.as_posix(), lang))
    return docs


def run_ingest(
    path: str | Path,
    *,
    collection: str | None = None,
    recreate: bool = False,
    prefer_bge: bool = False,
    contextualize: bool | None = None,
    contextualize_faq: bool = False,
    doc_type: DocTypeOpt = "auto",
    settings: Settings | None = None,
    llm: Any | None = None,
) -> dict[str, Any]:
    """
    Full ingestion pipeline. Returns stats dict.

    FAQ documents:
      - Atomic Q+A chunks (never split Q from A without repeating Q)
      - LLM contextualization **off by default** (``contextualize_faq=False``)
      - Raw Q/A preserved in chunk text and metadata

    Non-FAQ documents use the generic semantic/CJK chunker.
    """
    cfg = settings or get_settings()
    docs = load_documents(path)
    all_chunks: list[DocumentChunk] = []
    faq_chunks: list[DocumentChunk] = []
    generic_chunks: list[DocumentChunk] = []
    skipped_malformed = 0
    faq_files = 0
    generic_files = 0
    warnings: list[str] = []

    for text, source, lang in docs:
        title = Path(source).stem.replace("_", " ").title()
        # Front-matter doc_type if present
        fm_doc_type = None
        if text.lstrip().startswith("---"):
            from app.ingestion.faq_parser import parse_front_matter

            fm, _ = parse_front_matter(text)
            fm_doc_type = fm.get("doc_type") or fm.get("type")

        resolved = resolve_doc_type(
            explicit_doc_type=fm_doc_type,
            cli_doc_type=doc_type if doc_type != "auto" else None,
            text=text,
        )

        if resolved == "faq":
            faq_files += 1
            chunks, parse_result = chunk_faq_document(
                text,
                source=source,
                language=lang,
                title=title,
                explicit_meta={
                    "language": lang,
                },
            )
            skipped_malformed += parse_result.skipped_malformed
            warnings.extend(parse_result.warnings)
            if not chunks:
                warnings.append(f"{source}: FAQ mode produced zero chunks")
            faq_chunks.extend(chunks)
            all_chunks.extend(chunks)
        else:
            generic_files += 1
            chunks = chunk_document(
                text,
                language=lang,
                source=source,
                title=title,
                category=Path(source).stem,
                prefer_bge_m3=prefer_bge,
            )
            for i, c in enumerate(chunks):
                meta = dict(c.metadata or {})
                meta.setdefault("doc_type", "generic")
                chunks[i] = c.model_copy(update={"metadata": meta})
            generic_chunks.extend(chunks)
            all_chunks.extend(chunks)

    # Contextualization: default ON for generic (cfg), OFF for FAQ unless contextualize_faq
    if contextualize is None:
        contextualize = cfg.contextualize

    final_chunks: list[DocumentChunk] = []
    if faq_chunks:
        if contextualize_faq:
            # Still keep raw Q/A in metadata; contextual_text may be enriched
            enriched = contextualize_chunks(faq_chunks, llm=llm, settings=cfg)
            final_chunks.extend(enriched)
        else:
            # Ensure contextual_text is raw atomic text
            for c in faq_chunks:
                final_chunks.append(
                    c.model_copy(update={"contextual_text": c.contextual_text or c.text})
                )
    if generic_chunks:
        if contextualize:
            final_chunks.extend(
                contextualize_chunks(generic_chunks, llm=llm, settings=cfg)
            )
        else:
            for c in generic_chunks:
                title = str(c.metadata.get("title") or "")
                final_chunks.append(
                    c.model_copy(
                        update={
                            "contextual_text": f"{title}: {c.text}" if title else c.text
                        }
                    )
                )

    n = upsert_chunks(
        final_chunks,
        collection=collection or cfg.qdrant_collection,
        recreate=recreate,
        prefer_bge=prefer_bge,
        settings=cfg,
    )
    langs = sorted({str(c.metadata.get("language")) for c in final_chunks if c.metadata})
    logger.info(
        "Ingested %s points (faq_files=%s generic_files=%s skipped_malformed=%s); languages=%s",
        n,
        faq_files,
        generic_files,
        skipped_malformed,
        langs,
    )
    return {
        "points": n,
        "chunks": len(final_chunks),
        "faq_chunks": len(faq_chunks),
        "generic_chunks": len(generic_chunks),
        "documents": len(docs),
        "faq_files": faq_files,
        "generic_files": generic_files,
        "skipped_malformed": skipped_malformed,
        "languages": langs,
        "collection": collection or cfg.qdrant_collection,
        "warnings": warnings,
        "chunk_schema_version_faq": "faq_atomic_v1",
    }
