#!/usr/bin/env python3
"""Preview semantic/CJK chunks for sample (or arbitrary) documents."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root on path when run as script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingestion.chunking import chunk_files  # noqa: E402


def collect_paths(input_dir: Path) -> list[str]:
    paths: list[Path] = []
    for ext in ("*.md", "*.txt", "*.markdown"):
        paths.extend(sorted(input_dir.rglob(ext)))
    return [str(p) for p in paths]


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview multilingual chunks")
    parser.add_argument("--input", type=Path, required=True, help="Docs root directory")
    parser.add_argument("--out", type=Path, required=True, help="JSON output path")
    parser.add_argument(
        "--prefer-bge-m3",
        action="store_true",
        help="Try to load BGE-M3 (downloads weights if missing)",
    )
    args = parser.parse_args()

    paths = collect_paths(args.input)
    if not paths:
        print(f"No documents found under {args.input}", file=sys.stderr)
        return 1

    chunks = chunk_files(paths, prefer_bge_m3=args.prefer_bge_m3)
    sources_by_language: dict[str, list[str]] = defaultdict(list)
    languages_seen: set[str] = set()
    serializable = []
    for c in chunks:
        lang = str(c.metadata.get("language", "en"))
        src = str(c.metadata.get("source", ""))
        languages_seen.add(lang)
        if src and src not in sources_by_language[lang]:
            sources_by_language[lang].append(src)
        serializable.append(
            {
                "id": c.id,
                "text": c.text,
                "metadata": c.metadata,
            }
        )

    payload = {
        "languages_seen": sorted(languages_seen),
        "sources_by_language": dict(sources_by_language),
        "chunk_count": len(serializable),
        "chunks": serializable,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(serializable)} chunks; languages={payload['languages_seen']} → {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
