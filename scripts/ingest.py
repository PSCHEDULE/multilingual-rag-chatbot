#!/usr/bin/env python3
"""Ingest documents into Qdrant with Contextual Retrieval."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.pipeline import run_ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest support docs into Qdrant")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--prefer-bge", action="store_true")
    parser.add_argument(
        "--doc-type",
        choices=["auto", "faq", "generic"],
        default="auto",
        help="Document routing: auto (default), force faq, or force generic",
    )
    parser.add_argument(
        "--no-contextualize",
        action="store_true",
        help="Skip LLM contextualization for generic docs",
    )
    parser.add_argument(
        "--contextualize-faq",
        action="store_true",
        help="Enable LLM contextualization for FAQ chunks (off by default)",
    )
    args = parser.parse_args()

    # Default offline-friendly mock LLM unless key present
    if not (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("XAI_API_KEY")
    ):
        os.environ.setdefault("MOCK_LLM", "1")

    # Safety: never allow --recreate on operational hash baselines.
    from app.retrieval.qdrant_store import (  # noqa: E402
        PROTECTED_COLLECTIONS,
        assert_collection_write_allowed,
    )

    target = args.collection  # may be None → resolved inside run_ingest via settings
    if args.recreate:
        # If collection omitted, resolve default so we can block protected names.
        if not target:
            from app.config import get_settings  # noqa: E402

            target = get_settings().qdrant_collection
        try:
            assert_collection_write_allowed(target, recreate=True)
        except ValueError as exc:
            print(
                json.dumps(
                    {
                        "error": str(exc),
                        "protected_collections": sorted(PROTECTED_COLLECTIONS),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2

    stats = run_ingest(
        args.path,
        collection=args.collection,
        recreate=args.recreate,
        prefer_bge=args.prefer_bge,
        contextualize=not args.no_contextualize,
        contextualize_faq=args.contextualize_faq,
        doc_type=args.doc_type,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
