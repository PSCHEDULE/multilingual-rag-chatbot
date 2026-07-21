"""M7 gate policy: faithfulness hard; answer_relevancy diagnostic by default."""

from __future__ import annotations

from app.eval.run_ragas import (
    INITIAL,
    active_hard_gates,
    check_thresholds,
    diagnostic_threshold_notes,
    run_mock,
)


def _report(faithfulness: float | None, answer_relevancy: float | None) -> dict:
    metrics: dict[str, float] = {}
    if faithfulness is not None:
        metrics["faithfulness"] = faithfulness
    if answer_relevancy is not None:
        metrics["answer_relevancy"] = answer_relevancy
    return {"metrics": metrics}


def test_low_faithfulness_fails() -> None:
    report = _report(faithfulness=0.50, answer_relevancy=0.99)
    fails = check_thresholds(report, INITIAL)
    assert any(f.startswith("faithfulness:") for f in fails)
    assert not any("answer_relevancy" in f for f in fails)


def test_low_answer_relevancy_alone_does_not_fail_by_default() -> None:
    report = _report(faithfulness=0.90, answer_relevancy=0.10)
    fails = check_thresholds(report, INITIAL)
    assert fails == []
    notes = diagnostic_threshold_notes(report, INITIAL)
    assert any("answer_relevancy" in n and "diagnostic" in n for n in notes)


def test_low_answer_relevancy_fails_when_explicit_ar_gate_enabled() -> None:
    report = _report(faithfulness=0.90, answer_relevancy=0.10)
    fails = check_thresholds(report, INITIAL, gate_answer_relevancy=True)
    assert any(f.startswith("answer_relevancy:") for f in fails)
    # With AR gated, it is not merely diagnostic
    notes = diagnostic_threshold_notes(report, INITIAL, gate_answer_relevancy=True)
    assert not any(n.startswith("answer_relevancy:") for n in notes)


def test_passing_metrics_pass() -> None:
    report = _report(faithfulness=0.90, answer_relevancy=0.90)
    assert check_thresholds(report, INITIAL) == []
    assert check_thresholds(report, INITIAL, gate_answer_relevancy=True) == []


def test_metrics_reporting_still_includes_answer_relevancy() -> None:
    """Threshold tables and mock reports still expose answer_relevancy."""
    assert "answer_relevancy" in INITIAL
    assert "faithfulness" in INITIAL
    # Default hard gates exclude AR
    hard = active_hard_gates(INITIAL)
    assert "faithfulness" in hard
    assert "answer_relevancy" not in hard
    # Optional gate includes AR
    hard_ar = active_hard_gates(INITIAL, gate_answer_relevancy=True)
    assert "answer_relevancy" in hard_ar

    rows = [
        {
            "id": "t1",
            "language": "en",
            "question": "How do I get a refund?",
            "ground_truth": "Request a refund within 14 days.",
        }
    ]
    report = run_mock(rows)
    metrics = report.get("metrics") or {}
    assert "answer_relevancy" in metrics
    assert metrics["answer_relevancy"] is not None
    assert "faithfulness" in metrics
