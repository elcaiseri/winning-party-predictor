# Test 2 — A winning-party predictor over retrieved precedent

You are given a small agent that reads a U.S. Supreme Court case (its
facts) and predicts **which party the Court ruled for** — `petitioner`
or `respondent` — grounded in similar past cases it retrieves. The
precedents are real (opinion text from the Caselaw Access Project,
outcomes from the Supreme Court Database; see `data/SOURCE.md`), indexed
in a **ChromaDB** vector store seeded with a local **sentence-transformers**
embedding model (servable via vLLM in production).

The starter runs, but it is naive and the grounding is broken. Your job
is to make it a correct, honestly-evaluated classifier.

**Time budget:** 90 minutes total, with the last 15 reserved for a
walkthrough of your design decisions.

You may use any AI assistant during the session. We are evaluating your
design judgment, not whether you can type fast.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package
management; dependencies are pinned in `uv.lock`.

```bash
cd 02-critic-and-revise-loop
uv sync                        # creates .venv from the lock file

export OPENAI_API_KEY=...       # the classifier LLM
uv run python build_index.py          # seeds the ChromaDB index (once, local embeddings)
uv run python tests/sample_queries.py
```

Embeddings run locally via sentence-transformers, so `build_index.py`
needs no API key. (To serve the embedding model with vLLM instead, start
a vLLM server and set `EMBED_BASE_URL` to its OpenAI-compatible `/v1`
URL.) The `OPENAI_API_KEY` is provided for the session, temporary, and
revoked afterward — a shared, limited budget, so spending counts. If it
is unset, a small offline mock LLM runs so the harness still executes —
plumbing only; build and evaluate on OpenAI.

`tests/sample_queries.py` runs the predictor over a set of held-out real
cases and prints each prediction plus an evaluation summary.

## What you must build

### Part 1 — Build

1. **Structured output with a reason.** The predictor currently asks the
   model for a single word and string-matches it out of free text. Make
   the model return a **structured output**: the winner as an **enum**
   (`petitioner` / `respondent`) **and** a short **`reasoning`** that
   explains *why*, grounded in the retrieved precedent. `llm.py`'s
   `complete` accepts a `response_format` argument.

2. **Evaluation.** `metrics.py` reports raw accuracy and nothing else.
   Make it meaningful: compare against a **baseline**, report
   **coverage**, and account for **class imbalance**. You should be able
   to say whether the predictor is actually beating the trivial baseline.

### Part 2 — Fix

The starter runs, but the predictions are not actually grounded — it
performs at chance. More than one thing in the pipeline is wrong, and not
all of them show up as a stack trace. Find them and fix them so the
predictions are genuinely grounded in the retrieved precedent.

A note on what "good" means here: predicting Supreme Court outcomes from
facts is genuinely hard, and the labels are imbalanced, so **a high
accuracy number is not the goal** — a *correct, honestly measured*
pipeline is. A candidate who reports "I beat the majority-class baseline
by a few points, and here's why retrieval helps" is exactly right.

### Part 3 — Discuss (walkthrough)

Once it works, we'll talk through how you'd make it a *real* system: how
you would produce and trust the `reasoning`, how you would move from a
single-shot prediction toward something that argues both sides before
concluding, and how you'd evaluate any of this at scale. Come with
opinions.

## What we are evaluating

- **RAG correctness.** Is retrieval returning relevant, useful context,
  and is the prediction actually grounded in it?
- **Structured output.** Enum + grounded reasoning, not parsed prose.
- **Evaluation literacy.** Baseline, class balance, coverage — do you
  measure the classifier the way an ML engineer should?
- **Debugging method.** How you diagnose *why* the pipeline underperforms,
  and how you verify your fixes, matters as much as the fixes themselves.

## Things to get familiar with before the interview

About two hours. None of it is optional.

### Reading (1.5 hours)

- **OpenAI — Structured Outputs** —
  <https://platform.openai.com/docs/guides/structured-outputs>
  How to constrain a response to a JSON schema / enum. You will use this.

- **OpenAI — Function calling** —
  <https://platform.openai.com/docs/guides/function-calling>
  How the model requests a tool and how results flow back into the
  conversation.

- **Chroma — embeddings & query** —
  <https://docs.trychroma.com/docs/embeddings/embedding-functions>
  How documents and queries get embedded and searched.

- **Chip Huyen, "Building LLM applications for production"** —
  <https://huyenchip.com/2023/04/11/llm-engineering.html>
  Read the sections on evaluation and cost.

### Hands-on warm-up (30 minutes)

Set up a tiny Chroma collection, add a few documents, and query it. Then
make one OpenAI call that returns a value constrained to an enum via
`response_format`. Walking in having done both once will save you time.

### What you do NOT need

- No specific agent framework. Plain Python is fine and preferred.
- No deep U.S.-law knowledge. Everything you need is in the data.

## Format of the session

- 10 min: read the starter code and the sample cases.
- ~25 min: Part 1 (structured output + reasoning, and evaluation).
- ~25 min: Part 2 (find and fix the bugs).
- ~15 min: Part 3 (discussion / walkthrough of design + improvements).
- 5 min: questions from you about the role.

## Output

Run `uv run python tests/sample_queries.py` at the end. Be ready to walk through
what was broken and how you found it, why your evaluation is trustworthy,
and what the predictor's accuracy means relative to the baseline.

Good luck.
