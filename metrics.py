"""
Evaluation for the winning-party predictor.

`evaluate` is called in tests/sample_queries.py over all sample cases.
Right now it reports raw accuracy and nothing else — which is not enough
to know whether the predictor is any good. Improve it.

Things worth measuring: a baseline to compare against (what does always
guessing the majority class score?), coverage (how many predictions were
actually parsed vs. None), and per-class performance (the classes are
imbalanced). Return whatever makes the result trustworthy.
"""

from __future__ import annotations

from typing import Any


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    # results: list of {"case_id", "predicted": str | None, "expected": str}
    # TODO: make this meaningful (baseline, coverage, per-class, ...).
    correct = sum(1 for r in results if r["predicted"] == r["expected"])
    return {"accuracy": correct / len(results) if results else 0.0}
