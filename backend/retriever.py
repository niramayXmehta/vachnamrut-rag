"""
backend/retriever.py
────────────────────
Direct passage retrieval — query the unified ChromaDB passage-level index
and return the top-k most relevant passages with full metadata.

Supports optional corpus filtering via the `sources` parameter:
    ["vachanamrut", "swamini_vaato"]  — both (default)
    ["vachanamrut"]                   — Vachanamrut only
    ["swamini_vaato"]                 — Swamini Vaato only
"""

from __future__ import annotations

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

def retrieve(
    query   : str,
    top_k   : int | None = None,
    sources : list[str] | None = None,
) -> list[dict]:
    """
    Embed the query and retrieve the top-k most relevant passages.

    Args:
        query   : the user's question (any language)
        top_k   : number of passages to return (defaults to config.TOP_K_PASSAGES)
        sources : corpus names to search — ["vachanamrut", "swamini_vaato"]
                  None or both → searches full collection (no filter).
                  Single entry → adds a ChromaDB where clause.
    """
    k          = top_k or config.TOP_K_PASSAGES
    collection = _get_collection()

    q_vec = encode_single(query).tolist()

    # Build where clause only when filtering to a single corpus
    all_sources = {"vachanamrut", "swamini_vaato"}
    active      = set(sources) if sources else all_sources
    where       = None
    if active != all_sources and len(active) == 1:
        where = {"source": {"$eq": list(active)[0]}}

    query_kwargs: dict = dict(
        query_embeddings = [q_vec],
        n_results        = k,
        include          = ["metadatas", "distances", "documents"],
    )
    if where:
        query_kwargs["where"] = where

    results   = collection.query(**query_kwargs)
    metadatas = results["metadatas"][0]  # type: ignore[index]
    distances = results["distances"][0]  # type: ignore[index]

    passages = []
    for meta, dist in zip(metadatas, distances):
        passages.append({
            # ── shared ────────────────────────────────────────────────────────
            "passage_en"      : str(meta.get("passage_en") or ""),
            "passage_gu"      : str(meta.get("passage_gu") or ""),
            "source"          : str(meta.get("source", "vachanamrut")),
            "passage_index"   : int(meta.get("passage_index", 0)),
            "cosine_distance" : round(float(dist), 4),
            # ── Vachanamrut ───────────────────────────────────────────────────
            "vachno"          : int(meta.get("vachno", 0)),
            "section_en"      : str(meta.get("section_en", "")),
            "section_gu"      : str(meta.get("section_gu", "")),
            "num_in_section"  : int(meta.get("num_in_section", 0)),
            "title_en"        : str(meta.get("title_en", "")),
            "title_gu"        : str(meta.get("title_gu", "")),
            "section_heading" : str(meta.get("section_heading", "")),
            # ── Swamini Vaato ─────────────────────────────────────────────────
            "prakaran"        : int(meta.get("prakaran", 0)),
            "vat_number"      : int(meta.get("vat_number", 0)),
        })

    return passages


def is_low_relevance(passages: list[dict]) -> bool:
    if not passages:
        return True
    return passages[0]["cosine_distance"] > config.RELEVANCE_THRESHOLD