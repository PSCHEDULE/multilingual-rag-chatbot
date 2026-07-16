"""Eval dataset integrity."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "app" / "eval" / "dataset" / "cross_lingual_v1.jsonl"


def test_cross_lingual_dataset_minimum() -> None:
    rows = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 12
    langs = {r["language"] for r in rows}
    assert langs >= {"ko", "en", "ja", "zh"}
    intents = {r["intent"] for r in rows}
    assert len(intents) >= 3
    for r in rows:
        assert r["question"].strip()
        assert r["ground_truth"].strip()
