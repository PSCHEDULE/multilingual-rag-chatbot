"""RAGAS evaluation runner with initial / release threshold tiers.

Live path (no --mock): generate answers via the app graph and score with **real RAGAS**
using the same LLM as M2/M4 — OpenAI **gpt-4o-mini** (`LLM_PROVIDER=openai`).

Stability principles:
- Scoring failures (timeouts, HTTP errors, exceptions) are **not** coerced to 0.0.
- Failed samples are marked ``evaluation_error`` with error type/message.
- Aggregate metrics average only successfully scored samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# Threshold tiers from plan M7
INITIAL = {"faithfulness": 0.78, "answer_relevancy": 0.75}
RELEASE = {"faithfulness": 0.82, "answer_relevancy": 0.78}

# Default RunConfig for live RAGAS (correctness over speed)
DEFAULT_TIMEOUT = 600
DEFAULT_MAX_WORKERS = 2
DEFAULT_MAX_RETRIES = 3
# Outer retries when a whole-sample score call fails
SAMPLE_MAX_ATTEMPTS = 3


class _EvalEventCounter(logging.Handler):
    """Capture TimeoutError / HTTP 429 signals from nested libraries."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.timeout_errors = 0
        self.http_429 = 0
        self.other_errors = 0

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        low = msg.lower()
        if "timeouterror" in low or "timeout" in low and "error" in low:
            if "timeouterror" in low or "timed out" in low or "timeout error" in low:
                self.timeout_errors += 1
        if "429" in msg or "rate limit" in low or "too many requests" in low:
            self.http_429 += 1
        if record.levelno >= logging.ERROR:
            self.other_errors += 1


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize eval dataset schemas.

    Supports:
    - legacy: language, ground_truth
    - onlybook: lang, reference (+ expected_key_points)
    """
    out = dict(row)
    if not out.get("language") and out.get("lang"):
        out["language"] = out["lang"]
    if not out.get("ground_truth"):
        out["ground_truth"] = (
            out.get("reference")
            or out.get("reference_answer")
            or ""
        )
    if not out.get("id"):
        out["id"] = out.get("source_qid") or "unknown"
    return out


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(_normalize_row(json.loads(line)))
    return rows


def _token_set(text: str) -> set[str]:
    import re

    return set(re.findall(r"[\w\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]+", text.lower()))


def proxy_faithfulness(answer: str, contexts: list[str], ground_truth: str) -> float:
    """Lightweight offline proxy (used only by --mock wiring path)."""
    if not answer.strip():
        return 0.0
    ctx = " ".join(contexts)
    base = _token_set(ground_truth) | _token_set(ctx)
    ans = _token_set(answer)
    if not ans:
        return 0.0
    overlap = len(ans & base) / max(len(ans), 1)
    return max(0.0, min(1.0, 0.55 + 0.45 * overlap))


def proxy_relevancy(answer: str, question: str) -> float:
    if not answer.strip():
        return 0.0
    q = _token_set(question)
    a = _token_set(answer)
    if not q or not a:
        return 0.5
    return max(0.0, min(1.0, 0.5 + 0.5 * (len(q & a) / max(len(q), 1))))


def _avg(xs: list[float]) -> float | None:
    return float(statistics.mean(xs)) if xs else None


def _is_valid_score(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v) and not math.isinf(v)


def run_mock(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Wiring gate: synthetic high scores so schema/thresholds path is exercised offline."""
    by_lang: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall_f: list[float] = []
    overall_r: list[float] = []
    samples: list[dict[str, Any]] = []
    for row in rows:
        answer = row["ground_truth"]
        f = max(
            0.9,
            proxy_faithfulness(answer, [row["ground_truth"]], row["ground_truth"]),
        )
        r = max(0.85, proxy_relevancy(answer, row["question"]))
        lang = row["language"]
        by_lang[lang]["faithfulness"].append(f)
        by_lang[lang]["answer_relevancy"].append(r)
        overall_f.append(f)
        overall_r.append(r)
        samples.append(
            {
                "id": row["id"],
                "language": lang,
                "question": row["question"],
                "answer": answer,
                "contexts": [row["ground_truth"]],
                "reference_answer": row.get("ground_truth") or "",
                "faithfulness": f,
                "answer_relevancy": r,
                "retries": 0,
                "status": "ok",
                "error": None,
            }
        )

    return {
        "mock": True,
        "scorer": "proxy_mock",
        "llm_model": "n/a",
        "metrics": {
            "faithfulness": _avg(overall_f),
            "answer_relevancy": _avg(overall_r),
        },
        "by_language": {
            lang: {
                "faithfulness": _avg(m["faithfulness"]),
                "answer_relevancy": _avg(m["answer_relevancy"]),
                "n": len(m["faithfulness"]),
                "n_ok": len(m["faithfulness"]),
                "n_error": 0,
            }
            for lang, m in sorted(by_lang.items())
        },
        "samples": samples,
        "n": len(rows),
        "n_ok": len(rows),
        "n_evaluation_error": 0,
        "event_counts": {"timeout_errors": 0, "http_429": 0, "other_errors": 0},
    }


def _build_ragas_llm_and_embeddings():
    """RAGAS judge + embeddings via OpenAI gpt-4o-mini / text-embedding-3-small."""
    from app.config import get_settings

    cfg = get_settings()
    api_key = cfg.resolved_llm_api_key()
    if not api_key:
        raise RuntimeError(
            "Live RAGAS requires OPENAI_API_KEY or LLM_API_KEY "
            f"(provider={cfg.llm_provider}, model={cfg.llm_model})"
        )

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=cfg.llm_model,
        api_key=api_key,
        base_url=cfg.llm_base_url,
        temperature=0.0,
        max_tokens=cfg.llm_max_tokens,
    )
    emb = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=cfg.llm_base_url,
    )
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb), cfg.llm_model


def _make_run_config() -> Any:
    from ragas.run_config import RunConfig

    return RunConfig(
        timeout=DEFAULT_TIMEOUT,
        max_workers=DEFAULT_MAX_WORKERS,
        max_retries=DEFAULT_MAX_RETRIES,
    )


def _score_one_sample(
    sample: dict[str, Any],
    *,
    ragas_llm: Any,
    ragas_emb: Any,
    max_attempts: int = SAMPLE_MAX_ATTEMPTS,
) -> dict[str, Any]:
    """
    Score a single sample with outer retries.

    On failure: return evaluation_error record (scores left null — never force 0.0).
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness

    record: dict[str, Any] = {
        "id": sample["id"],
        "language": sample["language"],
        "question": sample["question"],
        "answer": sample["answer"],
        "contexts": sample["contexts"],
        "reference_answer": sample.get("ground_truth") or sample.get("reference_answer") or "",
        "faithfulness": None,
        "answer_relevancy": None,
        "retries": 0,
        "status": "ok",
        "error": None,
        "error_type": None,
        "error_message": None,
    }

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        record["retries"] = attempt - 1
        try:
            ds = Dataset.from_dict(
                {
                    "question": [sample["question"]],
                    "answer": [sample["answer"]],
                    "contexts": [sample["contexts"]],
                    "ground_truth": [record["reference_answer"]],
                }
            )
            # Fresh metric instances per attempt (avoid stale LLM binding state)
            result = evaluate(
                ds,
                metrics=[Faithfulness(), AnswerRelevancy()],
                llm=ragas_llm,
                embeddings=ragas_emb,
                run_config=_make_run_config(),
                raise_exceptions=True,
                batch_size=1,
                show_progress=False,
            )
            df = result.to_pandas()
            f_raw = df["faithfulness"].iloc[0]
            r_raw = df["answer_relevancy"].iloc[0]

            if not _is_valid_score(f_raw) or not _is_valid_score(r_raw):
                raise ValueError(
                    f"Invalid/NaN metric values: faithfulness={f_raw!r}, "
                    f"answer_relevancy={r_raw!r}"
                )

            record["faithfulness"] = float(f_raw)
            record["answer_relevancy"] = float(r_raw)
            record["status"] = "ok"
            record["error"] = None
            record["error_type"] = None
            record["error_message"] = None
            logger.info(
                "scored sample id=%s attempt=%s f=%.3f r=%.3f",
                sample["id"],
                attempt,
                record["faithfulness"],
                record["answer_relevancy"],
            )
            return record
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            record["retries"] = attempt
            logger.warning(
                "score attempt failed id=%s attempt=%s/%s err=%s: %s",
                sample["id"],
                attempt,
                max_attempts,
                type(exc).__name__,
                exc,
            )
            # Brief backoff for rate limits
            msg = str(exc).lower()
            if "429" in str(exc) or "rate limit" in msg:
                time.sleep(min(30, 5 * attempt))
            else:
                time.sleep(min(10, 2 * attempt))

    # All attempts failed — mark evaluation_error (do NOT invent 0.0 scores)
    err_type = type(last_exc).__name__ if last_exc else "UnknownError"
    err_msg = str(last_exc) if last_exc else "unknown scoring failure"
    record["status"] = "evaluation_error"
    record["faithfulness"] = None
    record["answer_relevancy"] = None
    record["error"] = {
        "type": err_type,
        "message": err_msg,
    }
    record["error_type"] = err_type
    record["error_message"] = err_msg
    logger.error(
        "evaluation_error id=%s type=%s message=%s retries=%s",
        sample["id"],
        err_type,
        err_msg,
        record["retries"],
    )
    return record


def write_sample_details(
    samples: list[dict[str, Any]],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Persist structured per-sample evaluation details."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fieldnames = [
        "id",
        "language",
        "status",
        "question",
        "answer",
        "reference_answer",
        "contexts_json",
        "faithfulness",
        "answer_relevancy",
        "retries",
        "error_type",
        "error_message",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow(
                {
                    "id": s.get("id"),
                    "language": s.get("language"),
                    "status": s.get("status"),
                    "question": s.get("question"),
                    "answer": s.get("answer"),
                    "reference_answer": s.get("reference_answer") or s.get("ground_truth") or "",
                    "contexts_json": json.dumps(s.get("contexts") or [], ensure_ascii=False),
                    "faithfulness": s.get("faithfulness"),
                    "answer_relevancy": s.get("answer_relevancy"),
                    "retries": s.get("retries", 0),
                    "error_type": s.get("error_type")
                    or ((s.get("error") or {}) or {}).get("type"),
                    "error_message": s.get("error_message")
                    or ((s.get("error") or {}) or {}).get("message"),
                }
            )
    logger.info("Wrote sample details → %s and %s", json_path, csv_path)


def run_live(
    rows: list[dict[str, Any]],
    *,
    detail_json: Path | None = None,
    detail_csv: Path | None = None,
) -> dict[str, Any]:
    """Generate answers via graph and score with real RAGAS (gpt-4o-mini)."""
    if os.environ.get("MOCK_LLM", "").lower() in {"1", "true", "yes"}:
        logger.warning("MOCK_LLM is set — live answers will be mock (scores will suffer)")

    from app.config import get_settings
    from app.graph.workflow import run_turn
    from app.llm.client import get_llm_client

    cfg = get_settings()
    llm = get_llm_client(cfg)
    warnings: list[str] = []
    event_counter = _EvalEventCounter()
    logging.getLogger().addHandler(event_counter)
    logging.getLogger("ragas").addHandler(event_counter)
    logging.getLogger("httpx").addHandler(event_counter)
    logging.getLogger("openai").addHandler(event_counter)

    if cfg.mock_llm or type(llm).__name__ == "MockLLMClient":
        warnings.append(
            "Generation used MockLLMClient — set OPENAI_API_KEY for real gpt-4o-mini answers"
        )

    # --- Generation phase ---
    raw_samples: list[dict[str, Any]] = []
    for row in rows:
        try:
            state = run_turn(
                {
                    "messages": [{"role": "user", "content": row["question"]}],
                    "language": row["language"],
                },
                llm=llm,
            )
            answer = state.get("answer") or ""
            contexts = [d.get("text") or "" for d in (state.get("documents") or [])]
            if not contexts:
                contexts = [row.get("ground_truth") or ""]
                warnings.append(f"empty_retrieval:{row['id']}")
            raw_samples.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "question": row["question"],
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": row.get("ground_truth") or "",
                    "reference_answer": row.get("ground_truth") or "",
                }
            )
            logger.info(
                "generated id=%s lang=%s answer_len=%s contexts=%s",
                row["id"],
                row["language"],
                len(answer),
                len(contexts),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("generation failed id=%s", row["id"])
            raw_samples.append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "question": row["question"],
                    "answer": "",
                    "contexts": [],
                    "ground_truth": row.get("ground_truth") or "",
                    "reference_answer": row.get("ground_truth") or "",
                    "generation_error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                }
            )

    # --- Scoring phase (per sample, with retries; errors ≠ 0.0) ---
    ragas_llm, ragas_emb, model_name = _build_ragas_llm_and_embeddings()
    scored: list[dict[str, Any]] = []
    for sample in raw_samples:
        if sample.get("generation_error"):
            scored.append(
                {
                    "id": sample["id"],
                    "language": sample["language"],
                    "question": sample["question"],
                    "answer": sample.get("answer") or "",
                    "contexts": sample.get("contexts") or [],
                    "reference_answer": sample.get("reference_answer") or "",
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "retries": 0,
                    "status": "evaluation_error",
                    "error": sample["generation_error"],
                    "error_type": sample["generation_error"].get("type"),
                    "error_message": sample["generation_error"].get("message"),
                }
            )
            continue

        logger.info(
            "scoring sample id=%s\n  Q: %s\n  A: %s\n  ref: %s\n  contexts: %s",
            sample["id"],
            sample["question"][:200],
            (sample.get("answer") or "")[:200],
            (sample.get("reference_answer") or "")[:200],
            len(sample.get("contexts") or []),
        )
        record = _score_one_sample(
            sample,
            ragas_llm=ragas_llm,
            ragas_emb=ragas_emb,
            max_attempts=SAMPLE_MAX_ATTEMPTS,
        )
        scored.append(record)

    # Persist detailed sample artifacts
    out_dir = Path("artifacts/eval")
    detail_json = detail_json or (out_dir / "samples_detail.json")
    detail_csv = detail_csv or (out_dir / "samples_detail.csv")
    write_sample_details(scored, json_path=detail_json, csv_path=detail_csv)

    # Aggregates over successful samples only
    by_lang_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"faithfulness": [], "answer_relevancy": []}
    )
    by_lang_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "n_ok": 0, "n_error": 0}
    )
    overall_f: list[float] = []
    overall_r: list[float] = []
    n_ok = 0
    n_err = 0

    for s in scored:
        lang = s["language"]
        by_lang_counts[lang]["n"] += 1
        if s.get("status") == "ok" and _is_valid_score(s.get("faithfulness")) and _is_valid_score(
            s.get("answer_relevancy")
        ):
            f = float(s["faithfulness"])
            r = float(s["answer_relevancy"])
            by_lang_scores[lang]["faithfulness"].append(f)
            by_lang_scores[lang]["answer_relevancy"].append(r)
            overall_f.append(f)
            overall_r.append(r)
            by_lang_counts[lang]["n_ok"] += 1
            n_ok += 1
        else:
            by_lang_counts[lang]["n_error"] += 1
            n_err += 1

    by_language: dict[str, Any] = {}
    for lang in sorted(set(by_lang_counts) | set(by_lang_scores)):
        by_language[lang] = {
            "faithfulness": _avg(by_lang_scores[lang]["faithfulness"]),
            "answer_relevancy": _avg(by_lang_scores[lang]["answer_relevancy"]),
            "n": by_lang_counts[lang]["n"],
            "n_ok": by_lang_counts[lang]["n_ok"],
            "n_error": by_lang_counts[lang]["n_error"],
        }

    report: dict[str, Any] = {
        "mock": False,
        "scorer": "ragas",
        "llm_model": model_name,
        "llm_provider": cfg.llm_provider,
        "run_config": {
            "timeout": DEFAULT_TIMEOUT,
            "max_workers": DEFAULT_MAX_WORKERS,
            "max_retries": DEFAULT_MAX_RETRIES,
            "sample_max_attempts": SAMPLE_MAX_ATTEMPTS,
        },
        "metrics": {
            "faithfulness": _avg(overall_f),
            "answer_relevancy": _avg(overall_r),
        },
        "metrics_note": (
            "Averages exclude samples with status=evaluation_error "
            "(failed scores are null, not 0.0)."
        ),
        "by_language": by_language,
        "samples": scored,
        "n": len(rows),
        "n_ok": n_ok,
        "n_evaluation_error": n_err,
        "event_counts": {
            "timeout_errors": event_counter.timeout_errors,
            "http_429": event_counter.http_429,
            "other_errors": event_counter.other_errors,
        },
        "detail_artifacts": {
            "json": str(detail_json),
            "csv": str(detail_csv),
        },
    }
    if warnings:
        report["warnings"] = sorted(set(warnings))

    # Detach handlers
    for name in ("", "ragas", "httpx", "openai"):
        logging.getLogger(name).removeHandler(event_counter)

    return report


def parse_fail_under(s: str | None, tier: dict[str, float]) -> dict[str, float]:
    thr = dict(tier)
    if not s:
        return thr
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        thr[k.strip()] = float(v.strip())
    return thr


def check_thresholds(report: dict[str, Any], thr: dict[str, float]) -> list[str]:
    fails = []
    metrics = report.get("metrics") or {}
    for k, min_v in thr.items():
        got = metrics.get(k)
        if got is None:
            fails.append(f"{k}: no successful scores (all evaluation_error or empty)")
            continue
        if float(got) < min_v:
            fails.append(f"{k}: {float(got):.3f} < {min_v}")
    return fails


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    p = argparse.ArgumentParser(description="Run RAGAS evaluation (gpt-4o-mini live)")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--output", type=Path, default=Path("artifacts/eval/report.json"))
    p.add_argument("--tier", choices=["initial", "release"], default="initial")
    p.add_argument(
        "--fail-under",
        type=str,
        default=None,
        help="e.g. faithfulness=0.78,answer_relevancy=0.75",
    )
    p.add_argument(
        "--detail-json",
        type=Path,
        default=None,
        help="Per-sample JSON detail path (default: artifacts/eval/samples_detail.json)",
    )
    p.add_argument(
        "--detail-csv",
        type=Path,
        default=None,
        help="Per-sample CSV detail path (default: artifacts/eval/samples_detail.csv)",
    )
    args = p.parse_args(argv)

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    tier = RELEASE if args.tier == "release" else INITIAL
    thr = parse_fail_under(args.fail_under, tier)

    detail_json = args.detail_json or args.output.parent / "samples_detail.json"
    detail_csv = args.detail_csv or args.output.parent / "samples_detail.csv"

    try:
        if args.mock:
            report = run_mock(rows)
            write_sample_details(
                report.get("samples") or [],
                json_path=detail_json,
                csv_path=detail_csv,
            )
            report["detail_artifacts"] = {"json": str(detail_json), "csv": str(detail_csv)}
        else:
            report = run_live(rows, detail_json=detail_json, detail_csv=detail_csv)
    except Exception as exc:  # noqa: BLE001
        report = {
            "mock": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "tier": args.tier,
            "thresholds": thr,
            "metrics": {},
            "by_language": {},
            "n": len(rows),
            "n_ok": 0,
            "n_evaluation_error": len(rows),
            "passed": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Wrote {args.output}", file=sys.stderr)
        print(f"LIVE_EVAL_ERROR: {exc}", file=sys.stderr)
        return 2

    report["thresholds"] = thr
    report["tier"] = args.tier
    report["passed"] = len(check_thresholds(report, thr)) == 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")
    print(
        f"Detail artifacts: {report.get('detail_artifacts')}",
        file=sys.stderr,
    )
    print(
        f"n_ok={report.get('n_ok')} n_evaluation_error={report.get('n_evaluation_error')} "
        f"events={report.get('event_counts')}",
        file=sys.stderr,
    )

    fails = check_thresholds(report, thr)
    if fails:
        print("FAIL thresholds:", "; ".join(fails), file=sys.stderr)
        return 1
    print("PASS thresholds", thr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
