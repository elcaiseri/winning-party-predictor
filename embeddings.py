"""
Local embedding function for ChromaDB.

The vector index is embedded with a sentence-transformers model that runs
locally (no API key), so retrieval is deterministic and reproducible.
The default is `all-MiniLM-L6-v2` (384-dim).

In production the same model is served by **vLLM**, which exposes an
OpenAI-compatible `/v1/embeddings` endpoint. Point the code at it by
setting `EMBED_BASE_URL` (e.g. `http://localhost:8000/v1`); otherwise the
model is loaded and run in-process via sentence-transformers.

Env:
  EMBED_MODEL     — model name (default all-MiniLM-L6-v2).
  EMBED_BASE_URL  — if set, use this OpenAI-compatible endpoint (vLLM).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from dotenv import load_dotenv

# Load EMBED_MODEL / EMBED_BASE_URL from a local .env if present.
load_dotenv(Path(__file__).resolve().parent / ".env")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BASE_URL = os.environ.get("EMBED_BASE_URL")


class LocalEmbeddingFunction(EmbeddingFunction):
    """Deterministic sentence-transformers embeddings (local or vLLM-served)."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("EMBED_MODEL", DEFAULT_MODEL)
        self._encoder = None  # lazily loaded

    def _encode_remote(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self._model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{EMBED_BASE_URL.rstrip('/')}/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
        rows = sorted(payload["data"], key=lambda r: r.get("index", 0))
        return [[float(x) for x in r["embedding"]] for r in rows]

    def _encode_local(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self._model)
        vecs = self._encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in v] for v in vecs]

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        return (
            self._encode_remote(texts) if EMBED_BASE_URL else self._encode_local(texts)
        )

    @staticmethod
    def name() -> str:
        return "local-sentence-transformers"

    def get_config(self) -> dict[str, Any]:
        return {"model": self._model}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LocalEmbeddingFunction":
        return LocalEmbeddingFunction(config.get("model", DEFAULT_MODEL))
