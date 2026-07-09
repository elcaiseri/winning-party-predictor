"""
Evaluation for the winning-party predictor.

`evaluate` is called in tests/sample_queries.py over all sample cases.
Beyond raw accuracy it reports what makes the number trustworthy:

- coverage: fraction of cases where a prediction was actually produced
  (unparsed / None predictions count as wrong in accuracy).
- majority_baseline: accuracy of always guessing the most common true
  label — the number the predictor has to beat.
- per_class precision/recall/F1 and balanced accuracy, since the classes
  are imbalanced in the corpus.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def evaluate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Score predictions against expected labels.

    Args:
        results: list of {"case_id", "predicted": str | None, "expected": str}.

    Returns:
        Dict with accuracy, coverage, majority baseline, balanced accuracy,
        and per-class precision/recall/F1/support.
    """
    n = len(results)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "coverage": 0.0}

    expected_counts = Counter(r["expected"] for r in results)
    labels = sorted(expected_counts)

    covered = [r for r in results if r["predicted"] is not None]
    correct = sum(1 for r in results if r["predicted"] == r["expected"])

    per_class: dict[str, dict[str, float | int]] = {}
    recalls: list[float] = []
    for label in labels:
        tp = sum(
            1 for r in results if r["predicted"] == label and r["expected"] == label
        )
        fp = sum(
            1 for r in results if r["predicted"] == label and r["expected"] != label
        )
        fn = sum(
            1 for r in results if r["predicted"] != label and r["expected"] == label
        )
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        recalls.append(recall)
        per_class[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": expected_counts[label],
        }

    majority_label, majority_count = expected_counts.most_common(1)[0]
    return {
        "n": n,
        "accuracy": round(correct / n, 3),
        "coverage": round(len(covered) / n, 3),
        "accuracy_when_predicted": round(
            sum(1 for r in covered if r["predicted"] == r["expected"]) / len(covered), 3
        )
        if covered
        else 0.0,
        "majority_baseline": round(majority_count / n, 3),
        "majority_label": majority_label,
        "balanced_accuracy": round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        "per_class": per_class,
    }
