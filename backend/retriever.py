"""
backend/retriever.py
────────────────────
Direct passage retrieval — query the ChromaDB passage-level index and
return the top-k most relevant passages with full metadata.

The coarse retrieval → passage splitting → re-ranking flow has been
replaced with a single ChromaDB search against ~1,800 passage vectors.
Each result already carries its source discourse metadata.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

import config
from backend.embedder import encode_single

# ─── CHROMA CLIENT (singleton) ────────────────────────────────────────────────

_collection: Collection | None = None


def _get_collection() -> Collection:
    global _collection
    if _collection is None:
        print(f"[retriever] Connecting to ChromaDB at {config.CHROMA_PATH} ...")
        client      = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        _collection = client.get_collection(name=config.COLLECTION_NAME)
        print(f"[retriever] Collection '{config.COLLECTION_NAME}' ready "
              f"({_collection.count()} passages).")
    return _collection


# ─── RETRIEVAL ────────────────────────────────────────────────────────────────

def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed the query and retrieve the top-k most relevant passages directly.

    Args:
        query : the user's question (any language)
        top_k : number of passages to return (defaults to config.TOP_K_PASSAGES)

    Returns:
        List of passage dicts, each containing:
            passage_en      : str   — English passage text
            passage_gu      : str   — Gujarati passage text
            vachno          : int
            section_en      : str
            section_gu      : str
            num_in_section  : int
            title_en        : str
            title_gu        : str
            section_heading : str
            passage_index   : int
            cosine_distance : float — lower is more similar
    """
    k          = top_k or config.TOP_K_PASSAGES
    collection = _get_collection()

    # Embed query
    q_vec = encode_single(query).tolist()

    # Query ChromaDB — returns passage-level results directly
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=k,
        include=["metadatas", "distances", "documents"],
    )

    metadatas = results["metadatas"][0]  # type: ignore[index]
    distances = results["distances"][0]  # type: ignore[index]

    passages = []
    for meta, dist in zip(metadatas, distances):
        passages.append({
            "passage_en"      : str(meta.get("passage_en") or ""),
            "passage_gu"      : str(meta.get("passage_gu") or ""),
            "vachno"          : meta["vachno"],
            "section_en"      : meta["section_en"],
            "section_gu"      : meta.get("section_gu", ""),
            "num_in_section"  : meta["num_in_section"],
            "title_en"        : meta.get("title_en", ""),
            "title_gu"        : meta.get("title_gu", ""),
            "section_heading" : meta.get("section_heading", ""),
            "passage_index"   : meta.get("passage_index", 0),
            "cosine_distance" : round(float(dist), 4),
        })

    return passages


def is_low_relevance(passages: list[dict]) -> bool:
    """
    Return True if even the best passage match is above the relevance threshold.
    """
    if not passages:
        return True
    return passages[0]["cosine_distance"] > config.RELEVANCE_THRESHOLD
