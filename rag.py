"""Embeddings + cosine-similarity retrieval.

Uses Nebius AI's bge-multilingual-gemma2 embedding model via
OpenAI-compatible API for high-quality Arabic + English embeddings.

Chunk embeddings are cached in memory so they are computed only ONCE
per document, not on every question.
"""
from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1/"
NEBIUS_EMBED_MODEL = os.getenv("NEBIUS_EMBED_MODEL", "Qwen/Qwen3-Embedding-8B")

_client = None
# Cache: maps a hash of the chunk list to its embedding matrix
_embed_cache: dict[int, np.ndarray] = {}


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY not set")
        _client = OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)
    return _client


def _embed_raw(texts: list[str]) -> np.ndarray:
    """Call Nebius embed API in batches."""
    client = _get_client()
    all_vecs: list[list[float]] = []
    # Batch in groups of 96 to stay safely under limits
    batch_size = 96
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client.embeddings.create(
            model=NEBIUS_EMBED_MODEL,
            input=batch,
        )
        all_vecs.extend([d.embedding for d in resp.data])
    arr = np.array(all_vecs, dtype=np.float32)
    # L2-normalize so dot product = cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    arr /= norms
    return arr


def embed(texts: list[str], use_cache: bool = True) -> np.ndarray:
    """Embed texts, using an in-memory cache to avoid redundant API calls."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    key = hash(tuple(texts))
    if use_cache and key in _embed_cache:
        return _embed_cache[key]
    vecs = _embed_raw(texts)
    if use_cache:
        _embed_cache[key] = vecs
    return vecs


def cosine_topk(
    query: str, chunks: list[str], k: int = 5
) -> list[tuple[int, float]]:
    """Return [(chunk_index, cosine_similarity)] for the top-k chunks."""
    if not chunks:
        return []
    # Query embedding — small, no cache needed
    q = _embed_raw([query])[0]
    # Chunk embeddings — cached, only computed once per document
    M = embed(chunks, use_cache=True)
    sims = M @ q  # both sides are L2-normalized
    k = min(k, len(chunks))
    idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in idx]


def cosine_sim(a: str, b: str) -> float:
    v = _embed_raw([a, b])
    return float(v[0] @ v[1])
