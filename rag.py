"""Embeddings + cosine-similarity retrieval.

Uses Voyage AI's voyage-law-2 embedding model.
Chunk embeddings are cached in memory so they are computed only ONCE
per document, not on every question.
"""
from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
from models import require_api_key

load_dotenv()

VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-law-2")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "voyage-rerank-2.5")

_client = None
# Cache: maps a hash of the chunk list to its embedding matrix
_embed_cache: dict[int, np.ndarray] = {}


def _get_client():
    global _client
    if _client is None:
        import voyageai

        api_key = require_api_key("VOYAGE_API_KEY")
        _client = voyageai.Client(api_key=api_key)
    return _client


def _embed_raw(texts: list[str]) -> np.ndarray:
    """Call Voyage AI embed API in batches with rate limit handling."""
    import time
    client = _get_client()
    all_vecs: list[list[float]] = []
    # Voyage has strict rate limits, so we use a smaller batch size and delay
    batch_size = 48
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(5):
            try:
                resp = client.embed(
                    batch,
                    model=VOYAGE_EMBED_MODEL,
                    input_type="document"
                )
                all_vecs.extend(resp.embeddings)
                time.sleep(1)  # small delay to prevent rapid RPM/TPM spikes
                break
            except Exception as e:
                # Catch rate limits (429) specifically
                if "RateLimitError" in str(type(e)) or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    print(f"Voyage rate limit hit, waiting {wait}s... (attempt {attempt+1})")
                    time.sleep(wait)
                else:
                    raise
        else:
            raise RuntimeError("Voyage embedding failed after max retries due to rate limits.")
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
    query: str, chunks: list[str], k: int = 5, use_reranker: bool = True
) -> list[tuple[int, float]]:
    """Return [(chunk_index, relevance_score)] for the top-k chunks.
    If use_reranker is True, gets top 20 by cosine, then reranks with Voyage.
    """
    if not chunks:
        return []
    
    # 1. First pass retrieval (vector search)
    q = _embed_raw([query])[0]
    M = embed(chunks, use_cache=True)
    sims = M @ q
    
    # Get top candidates for reranking
    num_candidates = min(20 if use_reranker else k, len(chunks))
    idx = np.argsort(-sims)[:num_candidates]
    
    if not use_reranker or len(chunks) == 1:
        # Just return cosine top k
        k = min(k, len(chunks))
        idx_k = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx_k]
    
    # 2. Second pass (Reranker)
    import time
    candidate_chunks = [chunks[i] for i in idx]
    client = _get_client()
    
    for attempt in range(4):
        try:
            resp = client.rerank(
                query=query,
                documents=candidate_chunks,
                model=VOYAGE_RERANK_MODEL,
                top_k=min(k, len(candidate_chunks))
            )
            break
        except Exception as e:
            if "RateLimitError" in str(type(e)) or "429" in str(e):
                wait = 5 * (attempt + 1)
                print(f"Voyage rerank rate limit hit, waiting {wait}s... (attempt {attempt+1})")
                time.sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Voyage reranking failed after max retries due to rate limits.")
    
    # resp.results contains objects with .index and .relevance_score
    # Map the candidate index back to the original chunk index
    final_results = []
    for r in resp.results:
        original_idx = int(idx[r.index])
        final_results.append((original_idx, float(r.relevance_score)))
        
    return final_results


def cosine_sim(a: str, b: str) -> float:
    v = _embed_raw([a, b])
    return float(v[0] @ v[1])
