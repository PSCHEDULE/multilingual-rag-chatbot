"""Post-rerank document pruning for generation context and sources."""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def prune_reranked_documents(
    documents: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    absolute_threshold: float | None = None,
    relative_threshold: float | None = None,
    max_keep: int = 3,
) -> list[dict[str, Any]]:
    """
    Phase-1 pruning after rerank (before LLM / sources event).

    - Always keep Top-1.
    - Keep Top-2 / Top-3 only if score >= absolute_threshold AND
      score >= Top-1 * relative_threshold (default 0.05 / 5%).
    - Keep at most ``max_keep`` documents (default 3).
    - Lower ranks are dropped.
    """
    cfg = settings or get_settings()
    abs_thr = (
        absolute_threshold
        if absolute_threshold is not None
        else float(cfg.prune_absolute_threshold)
    )
    rel_thr = (
        relative_threshold
        if relative_threshold is not None
        else float(cfg.prune_relative_threshold)
    )

    n_in = len(documents)
    if n_in == 0:
        logger.info("prune_docs candidates=0 kept=0")
        return []

    # Assume already score-desc sorted (rerank / multi-hop sort).
    top1_score = float(documents[0].get("score") or 0.0)
    kept: list[dict[str, Any]] = [documents[0]]
    pruned_reasons: list[str] = []

    for rank, doc in enumerate(documents[1:], start=1):
        score = float(doc.get("score") or 0.0)
        if rank >= max_keep:
            pruned_reasons.append(f"rank={rank} beyond_max_keep={max_keep}")
            continue
        # Top-2 / Top-3 only when both thresholds pass
        if score < abs_thr:
            pruned_reasons.append(
                f"rank={rank} score={score:.6f} below_absolute_threshold={abs_thr}"
            )
            continue
        rel_floor = top1_score * rel_thr
        if score < rel_floor:
            pruned_reasons.append(
                f"rank={rank} score={score:.6f} below_relative_threshold "
                f"(need>={rel_floor:.6f} = top1*{rel_thr})"
            )
            continue
        kept.append(doc)

    logger.info(
        "prune_docs candidates=%s kept=%s top1_score=%.6f abs_thr=%.4f rel_thr=%.4f "
        "pruned=%s",
        n_in,
        len(kept),
        top1_score,
        abs_thr,
        rel_thr,
        len(pruned_reasons),
    )
    for reason in pruned_reasons:
        logger.info("prune_docs drop %s", reason)

    return kept
