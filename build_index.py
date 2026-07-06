"""
Build the ChromaDB vector index of precedent cases from data/cases.json.

Run this ONCE during setup:

    python build_index.py

It seeds a persistent Chroma collection under ./chroma_db with embeddings
from a local sentence-transformers model (no API key). Re-running
rebuilds the index from scratch.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb

from embeddings import DEFAULT_MODEL, LocalEmbeddingFunction

ROOT = Path(__file__).parent
CASES_PATH = ROOT / "data" / "cases.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "us_case_law"
INDEX_MODEL = DEFAULT_MODEL  # the model the corpus is embedded with


def _document(case: dict) -> str:
    """The text that gets embedded for semantic search over a case."""
    return (
        f"{case['name']} ({case['citation']})\n"
        f"Area: {case['issue_area']}\n"
        f"Opinion: {case['text']}"
    )


def main() -> None:
    cases = json.loads(CASES_PATH.read_text())
    docs = [_document(c) for c in cases]

    # Embed the corpus with the local sentence-transformers model.
    embed = LocalEmbeddingFunction(INDEX_MODEL)
    vectors = embed(docs)

    print(f"Embedding {len(vectors)} cases with {INDEX_MODEL} to {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    col.add(
        ids=[c["id"] for c in cases],
        embeddings=vectors,
        documents=docs,
        metadatas=[
            {
                "name": c["name"],
                "citation": c["citation"],
                "issue_area": c["issue_area"],
                "winning_party": c["winning_party"],
                "year": c["year"] if c["year"] is not None else 0,
            }
            for c in cases
        ],
    )
    print(f"Indexed {col.count()} cases into '{COLLECTION}' with {INDEX_MODEL}")


if __name__ == "__main__":
    main()
