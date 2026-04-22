"""Embeddings + cosine-similarity retrieval.

Uses Voyage AI's voyage-law-2 embedding model.
Chunk embeddings are cached in memory so they are computed only ONCE
per document, not on every question.
"""
from __future__ import annotations

import os

import numpy as np
from dotenv import load_dotenv
import streamlit as st
from models import require_api_key, _get_db_connection

load_dotenv()

VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-law-2")
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")

_client = None


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
    # Voyage has strict rate limits, so we use a smaller batch size
    # 16 chunks = ~16,000 chars = ~4,000 tokens, safely below TPM limits
    batch_size = 16
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
                if hasattr(resp, "usage"):
                    from models import _add_tokens, tracked_sleep
                    _add_tokens("embed", getattr(resp.usage, "total_tokens", 0))
                tracked_sleep(1)  # small delay to prevent rapid RPM/TPM spikes
                break
            except Exception as e:
                # Catch rate limits (429) specifically
                if "RateLimitError" in str(type(e)) or "429" in str(e):
                    wait = 10 * (attempt + 1)
                    msg = f"Voyage embed rate limit hit, waiting {wait}s... (attempt {attempt+1})"
                    print(msg)
                    try:
                        st.warning(msg)
                    except:
                        pass
                    from models import tracked_sleep
                    tracked_sleep(wait)
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


def _doc_hash(texts: list[str]) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(VOYAGE_EMBED_MODEL.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
    return h.hexdigest()

def ensure_embedded(texts: list[str], filename: str) -> str:
    """Ensure texts are embedded and saved to DB. Returns document hash."""
    if not texts:
        return ""
        
    doc_hash = _doc_hash(texts)
    
    conn = _get_db_connection()
    with conn.cursor() as cur:
        # Check if already embedded
        cur.execute("SELECT 1 FROM document_chunks WHERE document_hash = %s LIMIT 1", (doc_hash,))
        if cur.fetchone():
            conn.close()
            return doc_hash
            
    # Not embedded, do it now using Voyage API
    try:
        st.toast(f"Embedding {len(texts)} chunks for {filename} via Voyage AI...", icon="⏳")
    except:
        pass
        
    vecs = _embed_raw(texts) # ndarray
    
    with conn.cursor() as cur:
        for i, (text, vec) in enumerate(zip(texts, vecs)):
            cur.execute("""
                INSERT INTO document_chunks (document_hash, chunk_index, chunk_text, embedding, filename)
                VALUES (%s, %s, %s, %s, %s)
            """, (doc_hash, i, text, vec.tolist(), filename))
    conn.commit()
    conn.close()
    
    try:
        st.toast(f"Saved {len(texts)} embeddings to PostgreSQL!", icon="✅")
    except:
        pass
    
    return doc_hash


def cosine_topk(
    query: str, chunks: list[str], filename: str, k: int = 5, use_reranker: bool = True
) -> list[tuple[int, float]]:
    """Return [(chunk_index, relevance_score)] for the top-k chunks.
    If use_reranker is True, gets top 20 from PostgreSQL via vector search, then reranks with Voyage.
    """
    if not chunks:
        return []
        
    # Ensure all chunks are embedded and mapped to a document_hash in the DB
    doc_hash = ensure_embedded(chunks, filename)
    
    # 1. First pass retrieval (PostgreSQL vector search)
    q_vec = _embed_raw([query])[0]
    num_candidates = min(20 if use_reranker else k, len(chunks))
    
    conn = _get_db_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT chunk_index, 1 - (embedding <=> %s::vector) AS similarity
            FROM document_chunks
            WHERE document_hash = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (q_vec.tolist(), doc_hash, q_vec.tolist(), num_candidates))
        results = cur.fetchall()
    conn.close()
    
    idx_and_sim = [(row[0], float(row[1])) for row in results]
    
    if not use_reranker or len(chunks) == 1:
        # Just return Postgres top k
        return idx_and_sim[:k]
    
    # 2. Second pass (Reranker)
    import time
    candidate_chunks = [chunks[i] for i, _ in idx_and_sim]
    client = _get_client()
    
    for attempt in range(4):
        try:
            resp = client.rerank(
                query=query,
                documents=candidate_chunks,
                model=VOYAGE_RERANK_MODEL,
                top_k=min(k, len(candidate_chunks))
            )
            if hasattr(resp, "usage"):
                from models import _add_tokens
                _add_tokens("embed", getattr(resp.usage, "total_tokens", 0))
            break
        except Exception as e:
            if "RateLimitError" in str(type(e)) or "429" in str(e):
                wait = 5 * (attempt + 1)
                msg = f"Voyage rerank rate limit hit, waiting {wait}s... (attempt {attempt+1})"
                print(msg)
                try:
                    st.warning(msg)
                except:
                    pass
                from models import tracked_sleep
                tracked_sleep(wait)
            else:
                raise
    else:
        raise RuntimeError("Voyage reranking failed after max retries due to rate limits.")
    
    # resp.results contains objects with .index and .relevance_score
    # Map the candidate index back to the original chunk index
    final_results = []
    for r in resp.results:
        original_idx = idx_and_sim[r.index][0]
        final_results.append((original_idx, float(r.relevance_score)))
        
    return final_results


def cosine_sim(a: str, b: str) -> float:
    v = _embed_raw([a, b])
    return float(v[0] @ v[1])
