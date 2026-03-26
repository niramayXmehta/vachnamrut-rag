"""
backend/reranker.py
───────────────────
Fine retrieval — re-rank a flat list of passages against the user query
using BGE-M3 embeddings and cosine similarity, then return the top-k.

Flow:
    1. Receive flat list of passages from passage_splitter.py
    2. Embed all passage_en texts in one batched call
    3. Compute cosine similarity between query vector and each passage vector
    4. Sort by similarity descending
    5. Return top-k passages with similarity score attached

Cosine similarity (higher = more relevant) is used here, as opposed to
cosine distance (lower = more relevant) used by ChromaDB. Both measure
the same thing — we just flip the sign for intuitive sorting.
"""

from __future__ import annotations

import numpy as np

import config
from backend.embedder import encode, encode_single

# ─── COSINE SIMILARITY ────────────────────────────────────────────────────────

def _cosine_similarity(query_vec: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query vector and a matrix
    of passage vectors.

    Args:
        query_vec    : shape (D,)
        passage_vecs : shape (N, D)

    Returns:
        np.ndarray of shape (N,) with similarity scores in [-1, 1]
    """
    # Normalise query
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

    # Normalise each passage vector
    norms        = np.linalg.norm(passage_vecs, axis=1, keepdims=True) + 1e-10
    p_normalised = passage_vecs / norms

    return p_normalised @ q_norm  # shape (N,)


# ─── RE-RANKING ───────────────────────────────────────────────────────────────

def rerank(
    query: str,
    passages: list[dict],
    top_k: int | None = None,
    query_vec: np.ndarray | None = None,
) -> list[dict]:
    """
    Re-rank passages by cosine similarity to the query and return top-k.

    Args:
        query     : the user's question (used to embed if query_vec not provided)
        passages  : flat list of passage dicts from passage_splitter.split_discourses()
        top_k     : number of passages to return (defaults to config.TOP_K_PASSAGES)
        query_vec : pre-computed query embedding — pass this in from pipeline.py
                    to avoid re-embedding the query a second time

    Returns:
        Top-k passage dicts sorted by similarity descending, each with an added
        'similarity' key (float, higher = more relevant).
    """
    if not passages:
        return []

    k = top_k or config.TOP_K_PASSAGES

    # Use pre-computed query vector if provided, otherwise embed now
    q_vec = query_vec if query_vec is not None else encode_single(query)

    # Extract English passage texts for embedding
    # Fall back to Gujarati if English is empty
    texts = [
        p["passage_en"] if p["passage_en"].strip() else p["passage_gu"]
        for p in passages
    ]

    # Embed all passages in one batched call
    passage_vecs = encode(texts)  # shape (N, 1024)

    # Compute similarities
    similarities = _cosine_similarity(q_vec, passage_vecs)  # shape (N,)

    # Attach similarity scores to passage dicts
    scored = []
    for passage, sim in zip(passages, similarities):
        scored.append({
            **passage,
            "similarity": round(float(sim), 4),
        })

    # Sort by similarity descending and return top-k
    scored.sort(key=lambda x: x["similarity"], reverse=True)

    return scored[:k]
