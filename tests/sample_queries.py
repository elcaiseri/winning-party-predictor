"""
Sample cases for the winning-party predictor.

Each case is a real Supreme Court case held out of the corpus: the agent
sees the facts and must predict which party won (petitioner or
respondent). The true label is in `expected_winner` — used only to
evaluate, never shown to the agent.

Run this file directly to see predictions and the evaluation summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import run_agent
from llm import LLM
from metrics import evaluate

CASES = json.loads((ROOT / "data" / "queries.json").read_text())


if __name__ == "__main__":
    llm = LLM()
    results = []
    for case in CASES:
        pred = run_agent(case, llm=llm)
        expected = case["expected_winner"]
        mark = "ok " if pred.predicted_winner == expected else "XX "
        print(f"{mark} {case['id'][:34].ljust(34)} predicted={str(pred.predicted_winner):11} expected={expected}")
        if pred.reasoning:
            print(f"      reasoning: {pred.reasoning[:160]}")
        results.append(
            {"case_id": case["id"], "predicted": pred.predicted_winner, "expected": expected}
        )

    print(f"\n[evaluation] {evaluate(results)}")
    print(f"[usage] total_tokens={llm.total_tokens()} total_cost=${llm.total_cost():.4f}")
