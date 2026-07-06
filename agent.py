"""
WinnerAgent — predicts which party the U.S. Supreme Court will rule for
(petitioner or respondent), grounded in retrieved precedent, and explains
why.

It retrieves similar past cases via a tool, then asks the model for the
predicted winner.

This starter runs, but it is naive and it has bugs. Your job:

  BUILD
  - The prediction is produced by asking the model for a word and
    string-matching it. Make the model return a **structured output**:
    the winner as an enum AND a short `reasoning` explaining why, grounded
    in the retrieved precedent. `llm.py`'s `complete` takes a
    `response_format` argument.
  - Make the evaluation in metrics.py meaningful (see that file).

  FIX
  - Run it and watch closely: what the retrieval tool returns, what its
    schema says, and how the model is being called and answered. Several
    things are wrong — some of them are not the kind of bug that shows up
    as a stack trace. Find them and fix them.

You may modify this file, tools.py, and metrics.py. Do not modify
data/, embeddings.py, or the labels.

Sample cases are in tests/sample_queries.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from llm import LLM, LLMResponse
from tools import TOOL_SCHEMAS, TOOL_REGISTRY

WINNING_PARTIES = ("petitioner", "respondent")
MAX_TOOL_ROUNDS = 4

CLASSIFY_SYSTEM_PROMPT = """You predict which party the U.S. Supreme Court will
rule for. Use the available tool to find similar past cases and reason from how
they came out. Then answer with a single word: petitioner or respondent.
"""


@dataclass
class Prediction:
    case_id: str
    predicted_winner: str | None
    reasoning: str | None = None
    total_cost: float = 0.0
    transcript: list[dict[str, Any]] = field(default_factory=list)


def _parse_winner(text: str) -> str | None:
    """Pull the predicted party out of the model's free-text answer."""
    lowered = text.lower()
    for party in WINNING_PARTIES:
        if party in lowered:
            return party
    return None


def run_agent(case: dict[str, Any], llm=None) -> Prediction:
    """Predict the winning party for a case."""
    llm = llm or LLM()
    cost_before = llm.total_cost()
    transcript: list[dict[str, Any]] = []

    messages = [
        {"role": "system", "content": CLASSIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Issue area: {case.get('issue_area', '?')}\n\nFacts:\n{case['facts']}",
        },
    ]

    resp: LLMResponse = llm.complete(messages, tools=TOOL_SCHEMAS, tool_choice="required")

    rounds = 0
    while resp.tool_calls and rounds < MAX_TOOL_ROUNDS:
        for tc in resp.tool_calls:
            result = TOOL_REGISTRY[tc.name](**tc.input)
            transcript.append({"step": "tool", "name": tc.name, "content": result})
            # hand the retrieved precedent back to the model
            messages.append({"role": "user", "content": json.dumps(result)})
        resp = llm.complete(messages, tools=TOOL_SCHEMAS, tool_choice="required")
        rounds += 1

    winner = _parse_winner(resp.text)
    transcript.append({"step": "answer", "content": resp.text})

    return Prediction(
        case_id=case.get("id", "?"),
        predicted_winner=winner,
        reasoning=None,
        total_cost=llm.total_cost() - cost_before,
        transcript=transcript,
    )
