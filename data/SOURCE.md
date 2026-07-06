# Corpus & test-set provenance

Everything is grounded in real cases, from two public sources:

- **Caselaw Access Project** (<https://static.case.law/>) — verbatim
  opinion text. U.S. Supreme Court opinions are in the public domain.
- **Supreme Court Database (SCDB)**, Washington University
  (<http://scdb.wustl.edu/>) — authoritative case metadata, including the
  **winning party** (`partyWinning`) and **issue area**.

## `cases.json` — the retrieval corpus

Landmark precedents, joined from both sources by citation. Each record:
`name`, `citation`, `year`, `issue_area` and `winning_party` (real SCDB
values), and `text` (a verbatim excerpt of the opinion, incl. a window
around the holding). This is the "known law" the agent retrieves from.

## `queries.json` — the held-out prediction task

Real cases, **held out of the corpus** and chosen to be lower-profile so
the model cannot simply recall the outcome. For each case:

- `expected_winner` is the real SCDB `partyWinning` label (petitioner or
  respondent).
- `facts` is a verbatim excerpt from the **opening** of the opinion
  (facts + procedural posture), truncated *before* the Court states its
  disposition, and leak-checked so it does not reveal who won at the
  Supreme Court.
- Each test case was selected so that a topically similar precedent
  exists in the corpus (verified with a local proxy embedding index), so
  retrieval genuinely has something relevant to find.

A small curated slice for an interview, not an authoritative dataset.
`build_index.py` seeds a ChromaDB index (`chroma_db/`) with embeddings
a local sentence-transformers model (see `embeddings.py`; servable via
vLLM in production).
