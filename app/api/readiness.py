"""Runtime readiness checks (Qdrant, collection dim, embedder).

Heavy embedder validation runs at **startup** only. ``GET /ready`` reuses the
startup snapshot and only performs lightweight config + Qdrant metadata checks.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings
from app.retrieval.config_guard import (
    EXPECTED_BGE_DIM,
    EXPECTED_HASH_DIM,
    RetrievalConfigError,
    collection_dense_dim,
    validate_prefer_bge_collection_pair,
)
from app.retrieval.embeddings import BGEEmbedder, OfflineHashEmbedder, get_dense_embedder
from app.retrieval.qdrant_store import get_qdrant_client, qdrant_reachable

logger = logging.getLogger(__name__)


def expected_collection_dim(prefer_bge: bool) -> int:
    return EXPECTED_BGE_DIM if prefer_bge else EXPECTED_HASH_DIM


def _validate_embedder_once(cfg: Settings) -> dict[str, Any]:
    """
    Load/validate embedder **once** (startup path only).

    Uses the canonical ``get_dense_embedder`` cache (no ``cache_clear``) so the
    startup guard, readiness snapshot, and request retrieval share one BGE
    instance. Does not call ``cache_clear`` on the hot path of /ready either.
    """
    if not cfg.prefer_bge:
        emb = OfflineHashEmbedder()
        return {
            "ok": True,
            "type": type(emb).__name__,
            "dim": emb.dim,
            "expected_dim": EXPECTED_HASH_DIM,
        }

    emb = get_dense_embedder(
        prefer_bge=True,
        model_name=cfg.embedding_model,
        require_bge=True,
    )
    ok_type = isinstance(emb, BGEEmbedder)
    dim = int(getattr(emb, "dim", 0) or 0)
    ok_dim = dim == EXPECTED_BGE_DIM
    result = {
        "ok": ok_type and ok_dim,
        "type": type(emb).__name__,
        "dim": dim,
        "expected_dim": EXPECTED_BGE_DIM,
    }
    if not ok_type:
        result["detail"] = f"embedder type {type(emb).__name__}, expected BGEEmbedder"
    if not ok_dim:
        result["detail"] = f"embedder dim={dim}, expected {EXPECTED_BGE_DIM}"
    return result


def _check_qdrant_collection_light(cfg: Settings) -> tuple[dict[str, Any], list[str]]:
    """Lightweight Qdrant connectivity + collection metadata (no embeddings)."""
    checks: dict[str, Any] = {}
    errors: list[str] = []

    try:
        reachable = qdrant_reachable(cfg)
        checks["qdrant_reachable"] = {"ok": reachable}
        if not reachable:
            errors.append("Qdrant is not reachable")
            checks["collection_exists"] = {"ok": False, "detail": "skipped (qdrant down)"}
            checks["collection_dim"] = {"ok": False, "detail": "skipped (qdrant down)"}
            return checks, errors
    except Exception as exc:  # noqa: BLE001
        checks["qdrant_reachable"] = {"ok": False, "detail": str(exc)}
        errors.append(f"qdrant: {exc}")
        checks["collection_exists"] = {"ok": False, "detail": "skipped (qdrant down)"}
        checks["collection_dim"] = {"ok": False, "detail": "skipped (qdrant down)"}
        return checks, errors

    try:
        client = get_qdrant_client(cfg)
        names = {c.name for c in client.get_collections().collections}
        exists = cfg.qdrant_collection in names
        checks["collection_exists"] = {
            "ok": exists,
            "name": cfg.qdrant_collection,
        }
        if not exists:
            errors.append(f"collection {cfg.qdrant_collection!r} not found")
            checks["collection_dim"] = {"ok": False, "detail": "collection missing"}
            return checks, errors

        dim = collection_dense_dim(client, cfg.qdrant_collection)
        expected = expected_collection_dim(cfg.prefer_bge)
        ok_dim = dim == expected
        checks["collection_dim"] = {
            "ok": ok_dim,
            "dim": dim,
            "expected_dim": expected,
        }
        if not ok_dim:
            errors.append(
                f"collection dim={dim}, expected {expected} "
                f"(prefer_bge={cfg.prefer_bge})"
            )
        try:
            info = client.get_collection(cfg.qdrant_collection)
            checks["collection_points"] = {
                "ok": True,
                "points_count": info.points_count,
            }
        except Exception as exc:  # noqa: BLE001
            checks["collection_points"] = {"ok": False, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        checks["collection_exists"] = {"ok": False, "detail": str(exc)}
        checks["collection_dim"] = {"ok": False, "detail": str(exc)}
        errors.append(f"collection check: {exc}")

    return checks, errors


def build_startup_readiness_snapshot(settings: Settings | None = None) -> dict[str, Any]:
    """
    One-time startup validation (may load BGE once).

    Stores embedder + collection state for reuse by lightweight ``check_readiness``.
    """
    cfg = settings or get_settings()
    checks: dict[str, Any] = {}
    errors: list[str] = []

    try:
        validate_prefer_bge_collection_pair(
            collection=cfg.qdrant_collection,
            prefer_bge=cfg.prefer_bge,
        )
        checks["config_pair"] = {"ok": True}
    except RetrievalConfigError as exc:
        checks["config_pair"] = {"ok": False, "detail": str(exc)}
        errors.append(str(exc))

    try:
        emb_check = _validate_embedder_once(cfg)
        checks["embedder"] = emb_check
        if not emb_check.get("ok"):
            errors.append(emb_check.get("detail") or "embedder validation failed")
    except Exception as exc:  # noqa: BLE001
        checks["embedder"] = {"ok": False, "detail": str(exc)}
        errors.append(f"embedder: {exc}")

    q_checks, q_errors = _check_qdrant_collection_light(cfg)
    checks.update(q_checks)
    errors.extend(q_errors)

    ready = (
        checks.get("config_pair", {}).get("ok")
        and checks.get("embedder", {}).get("ok")
        and checks.get("qdrant_reachable", {}).get("ok")
        and checks.get("collection_exists", {}).get("ok")
        and checks.get("collection_dim", {}).get("ok")
        and not errors
    )

    status = "ready" if ready else "not_ready"
    if cfg.prefer_bge and ready:
        status = "ok_bge_mode"
    elif not cfg.prefer_bge and checks.get("config_pair", {}).get("ok"):
        # Hash mode may start without Qdrant in some local setups; startup guard
        # historically only required config pair for process start.
        if ready:
            status = "ok_hash_mode"
        elif checks.get("embedder", {}).get("ok") and checks.get("config_pair", {}).get("ok"):
            if not checks.get("qdrant_reachable", {}).get("ok"):
                status = "ok_hash_embedder_qdrant_unreachable"
            else:
                status = "not_ready"
    elif cfg.prefer_bge and checks.get("embedder", {}).get("ok"):
        if not checks.get("qdrant_reachable", {}).get("ok"):
            status = "ok_bge_embedder_qdrant_unreachable"

    return {
        "ready": ready,
        "status": status,
        "prefer_bge": cfg.prefer_bge,
        "collection": cfg.qdrant_collection,
        "embedding_model": cfg.embedding_model,
        "expected_dense_dim": expected_collection_dim(cfg.prefer_bge),
        "checks": checks,
        "errors": list(errors),
        "embedder_dim": checks.get("embedder", {}).get("dim"),
        "collection_dim": checks.get("collection_dim", {}).get("dim"),
    }


def check_readiness(
    settings: Settings | None = None,
    *,
    startup_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Lightweight readiness probe for periodic /ready calls.

    * Does **not** call ``get_dense_embedder.cache_clear()``
    * Does **not** recreate or reload the BGE model
    * Does **not** run embedding or reranking inference
    * Reuses embedder validation from ``startup_report``
    * Optionally rechecks Qdrant connectivity and collection dense dim (metadata only)
    """
    cfg = settings or get_settings()
    checks: dict[str, Any] = {}
    errors: list[str] = []

    report: dict[str, Any] = {
        "ready": False,
        "prefer_bge": cfg.prefer_bge,
        "collection": cfg.qdrant_collection,
        "embedding_model": cfg.embedding_model,
        "expected_dense_dim": expected_collection_dim(cfg.prefer_bge),
        "checks": checks,
        "errors": errors,
        "startup": startup_report,
        "probe": "lightweight",
    }

    # 1) Config pair (cheap, no models)
    try:
        validate_prefer_bge_collection_pair(
            collection=cfg.qdrant_collection,
            prefer_bge=cfg.prefer_bge,
        )
        checks["config_pair"] = {"ok": True}
    except RetrievalConfigError as exc:
        checks["config_pair"] = {"ok": False, "detail": str(exc)}
        errors.append(str(exc))

    # 2) Embedder — from startup snapshot only (never reload on /ready)
    if startup_report and isinstance(startup_report.get("checks"), dict):
        emb = startup_report["checks"].get("embedder") or {}
        # Re-validate dim expectations against current mode
        expected = expected_collection_dim(cfg.prefer_bge)
        emb_ok = bool(emb.get("ok"))
        emb_dim = emb.get("dim")
        if emb_ok and emb_dim is not None and int(emb_dim) != expected:
            emb_ok = False
            errors.append(
                f"startup embedder dim={emb_dim} incompatible with current mode "
                f"(expected {expected}); restart required after config change"
            )
        # Mode flip without restart
        if startup_report.get("prefer_bge") != cfg.prefer_bge:
            emb_ok = False
            errors.append(
                "PREFER_BGE changed since startup; restart required for readiness"
            )
        if startup_report.get("collection") != cfg.qdrant_collection:
            # Collection change needs restart so startup snapshot matches
            errors.append(
                "QDRANT_COLLECTION changed since startup; restart required for readiness"
            )
            # still attach embedder status from startup
        checks["embedder"] = {
            "ok": emb_ok,
            "type": emb.get("type"),
            "dim": emb_dim,
            "expected_dim": expected,
            "source": "startup_snapshot",
        }
        if not emb_ok and not any("embedder" in e for e in errors):
            errors.append(emb.get("detail") or "embedder not validated at startup")
    else:
        checks["embedder"] = {
            "ok": False,
            "detail": "startup embedder snapshot missing; process may not have completed startup",
            "source": "startup_snapshot",
        }
        errors.append("startup embedder snapshot missing")

    # 3–4) Lightweight Qdrant recheck (metadata only)
    q_checks, q_errors = _check_qdrant_collection_light(cfg)
    checks.update(q_checks)
    errors.extend(q_errors)

    # 5) Startup snapshot attachment
    if startup_report is not None:
        checks["startup_guard"] = {
            "ok": startup_report.get("status")
            not in (None, "failed", "not_ready"),
            "status": startup_report.get("status"),
        }
        # failed startup should block ready
        if startup_report.get("status") == "failed":
            errors.append("startup retrieval checks failed")
            checks["startup_guard"]["ok"] = False
    else:
        checks["startup_guard"] = {
            "ok": False,
            "status": "not_recorded",
            "detail": "startup report not attached",
        }
        errors.append("startup report not attached")

    hard = (
        "config_pair",
        "embedder",
        "qdrant_reachable",
        "collection_exists",
        "collection_dim",
    )
    report["ready"] = all(checks.get(k, {}).get("ok") for k in hard) and not errors
    report["status"] = "ready" if report["ready"] else "not_ready"
    return report
