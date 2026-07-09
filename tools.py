"""
Tools the agent can call, backed by a ChromaDB vector index of real U.S.
Supreme Court precedents (built by build_index.py; provenance in
data/SOURCE.md).

Schemas are exposed in OpenAI function-calling format via TOOL_SCHEMAS,
and TOOL_REGISTRY maps tool names to the callables.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb

from embeddings import DEFAULT_MODEL, LocalEmbeddingFunction

ROOT = Path(__file__).parent
CASES_PATH = ROOT / "data" / "cases.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "us_case_law"

# Queries must be embedded with the SAME model the index was built with
# (build_index.py uses DEFAULT_MODEL); a different model means a different
# vector space and silently irrelevant results.
QUERY_EMBED_MODEL = DEFAULT_MODEL


@lru_cache(maxsize=1)
def _cases_by_id() -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in json.loads(CASES_PATH.read_text())}


@lru_cache(maxsize=1)
def _collection():
    if not CHROMA_DIR.exists():
        raise RuntimeError("Vector index not found. Run `python build_index.py` first.")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION)


@lru_cache(maxsize=2)
def _query_embedder(model: str) -> LocalEmbeddingFunction:
    return LocalEmbeddingFunction(model)


def _embed_query(text: str) -> list[float]:
    return _query_embedder(QUERY_EMBED_MODEL)([text])[0]


def search_precedents(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Return the `k` precedents most similar to `query`."""
    qvec = _embed_query(query)
    col = _collection()
    res = col.query(query_embeddings=[qvec], n_results=max(1, min(k, col.count())))
    out: list[dict[str, Any]] = []
    for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        full = _cases_by_id().get(cid, {})
        out.append(
            {
                "id": cid,
                "name": meta["name"],
                "citation": meta["citation"],
                "issue_area": meta["issue_area"],
                "winning_party": meta["winning_party"],
                "text": full.get("text", ""),
                "distance": round(dist, 4),
            }
        )
    return out


def get_case(case_id: str) -> dict[str, Any] | None:
    """Return the full record for a precedent id, or None."""
    return _cases_by_id().get(case_id)


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_precedents",
            "description": (
                "Semantic search over past Supreme Court cases. Returns the k "
                "most similar precedents for a fact pattern, each with its "
                "issue area, opinion text, and which party won."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Fact pattern or legal question."},
                    "k": {"type": "integer", "description": "Number of precedents to return (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_case",
            "description": "Fetch the full record (facts, holding, outcome) for a precedent by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string", "description": "Precedent id, e.g. 'terry-v-ohio'."},
                },
                "required": ["case_id"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "search_precedents": search_precedents,
    "get_case": get_case,
}
