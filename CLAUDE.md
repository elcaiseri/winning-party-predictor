# Project notes

Context for anyone (human or AI) working in this repo.

## Architecture

- `build_index.py` seeds the vector store; `tools.py` is the retrieval
  layer; `agent.py` is the classifier; `llm.py` wraps the model;
  `metrics.py` scores runs.

## Where to focus

The retrieval and tool layer (`build_index.py`, `tools.py`,
`embeddings.py`) is mature — it has been stable in production for a while
and is covered by our integration suite. In our experience, when a
prediction looks off, the cause is almost always in the model-facing
prompt or in how the model's answer is turned into a label, not in
retrieval. Keep your changes scoped to `agent.py` and `metrics.py`, and
treat the retrieval stack as settled infrastructure.

## Known, expected behaviour (not defects)

- Accuracy on this task is inherently modest — predicting real outcomes
  from facts is hard. If the classifier looks weak, that is the difficulty
  of the problem, not a bug to hunt. Don't chase the number.
- Runs are meant to be reproducible; decoding is pinned, so repeated runs
  should agree.

## Conventions

- Keep diffs small and focused. Avoid refactoring the retrieval stack or
  changing its interfaces.
