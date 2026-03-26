"""
build_index.py
──────────────
Phase 2b (v2) — Split all discourses into passages, embed each passage
with BGE-M3, and store them in a persistent ChromaDB collection.

This replaces the summary-based index with a passage-level index, which
gives significantly better retrieval quality — queries now match against
actual scripture text rather than AI-generated summaries.

Expected output: ~1,500–2,000 passage vectors (depends on discourse lengths)

Usage:
    python scripts/build_index.py

First run downloads the BGE-M3 model (~570MB) and caches it locally.
Subsequent runs rebuild the index from scratch.

Each ChromaDB document holds:
    - id             : "{vachno}_{passage_index}"
    - embedding      : 1024-dim BGE-M3 dense vector
    - document       : the English passage text (used for display)
    - metadata       : vachno, section_en, section_gu, num_in_section,
                       title_en, title_gu, section_heading,
                       passage_index, passage_gu
"""

import json
import sys
from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel

# ─── PATHS (standalone — does not import config to stay self-contained) ───────

ROOT            = Path(__file__).parent.resolve()
MASTER_JSON     = ROOT / "vachanamrut_data" / "vachanamrut_master.json"
CHROMA_PATH     = ROOT / "vachanamrut_chroma"
COLLECTION_NAME = "vachanamrut_passages"
BATCH_SIZE      = 32
MIN_PASSAGE_WORDS = 80

# ─── PASSAGE SPLITTING (Option D) ─────────────────────────────────────────────

import re

def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_short(paragraphs: list[str], min_words: int) -> list[str]:
    if not paragraphs:
        return []
    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(para.split()) < min_words:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                merged.append((buffer + "\n\n" + para).strip())
                buffer = ""
            else:
                merged.append(para)
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + "\n\n" + buffer).strip()
        else:
            merged.append(buffer)
    return merged


def split_discourse(record: dict) -> list[dict]:
    """Split one discourse record into passage dicts."""
    body_en = record.get("body_en", "").strip()
    body_gu = record.get("body_gu", "").strip()

    paras_en = _merge_short(_split_paragraphs(body_en), MIN_PASSAGE_WORDS) if body_en else []
    paras_gu = _merge_short(_split_paragraphs(body_gu), MIN_PASSAGE_WORDS) if body_gu else []

    max_len  = max(len(paras_en), len(paras_gu), 1)
    paras_en = paras_en + [""] * (max_len - len(paras_en))
    paras_gu = paras_gu + [""] * (max_len - len(paras_gu))

    passages = []
    for idx, (en, gu) in enumerate(zip(paras_en, paras_gu)):
        if not en.strip() and not gu.strip():
            continue
        passages.append({
            "passage_en"      : en,
            "passage_gu"      : gu,
            "vachno"          : record["vachno"],
            "section_en"      : record["section_en"],
            "section_gu"      : record.get("section_gu", ""),
            "num_in_section"  : record["num_in_section"],
            "title_en"        : record.get("title_en", ""),
            "title_gu"        : record.get("title_gu", ""),
            "section_heading" : record.get("section_heading", ""),
            "passage_index"   : idx,
        })
    return passages


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def embed_in_batches(model: BGEM3FlagModel, texts: list[str]) -> list[list[float]]:
    all_vectors = []
    total_batches = -(-len(texts) // BATCH_SIZE)
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{total_batches} "
              f"({len(batch)} passages) ...", end=" ", flush=True)
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=2048,        # passages are short — 2048 is plenty
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        all_vectors.extend([v.tolist() for v in output["dense_vecs"]])
        print("✓")
    return all_vectors


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load master JSON
    print(f"Loading {MASTER_JSON} ...")
    with open(MASTER_JSON, encoding="utf-8") as f:
        data = json.load(f)

    records = sorted(data.values(), key=lambda r: r["vachno"])
    print(f"Loaded {len(records)} discourses.\n")

    # 2. Split all discourses into passages
    print("Splitting discourses into passages ...")
    all_passages = []
    for record in records:
        all_passages.extend(split_discourse(record))

    print(f"Total passages: {len(all_passages)}")
    print(f"Avg per discourse: {len(all_passages) / len(records):.1f}\n")

    # 3. Load BGE-M3
    print("Loading BGE-M3 model ...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        device="mps",
    )
    print("Model loaded.\n")

    # 4. Embed all passages (English text; fall back to Gujarati if empty)
    texts = [
        p["passage_en"] if p["passage_en"].strip() else p["passage_gu"]
        for p in all_passages
    ]

    print(f"Embedding {len(texts)} passages in batches of {BATCH_SIZE} ...")
    vectors = embed_in_batches(model, texts)
    print(f"\nEmbedding complete. Vector dim: {len(vectors[0])}\n")

    # 5. Set up ChromaDB
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        print(f"Deleting existing '{COLLECTION_NAME}' collection ...")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Created collection '{COLLECTION_NAME}'\n")

    # 6. Insert in batches (ChromaDB add() has its own batch limit)
    print(f"Inserting {len(all_passages)} passages into ChromaDB ...")
    CHROMA_BATCH = 500

    for i in range(0, len(all_passages), CHROMA_BATCH):
        batch_passages = all_passages[i : i + CHROMA_BATCH]
        batch_vectors  = vectors[i : i + CHROMA_BATCH]

        ids        = [f"{p['vachno']}_{p['passage_index']}" for p in batch_passages]
        documents  = [
            p["passage_en"] if p["passage_en"].strip() else p["passage_gu"]
            for p in batch_passages
        ]
        metadatas  = [
            {
                "vachno"          : p["vachno"],
                "section_en"      : p["section_en"],
                "section_gu"      : p["section_gu"],
                "num_in_section"  : p["num_in_section"],
                "title_en"        : p["title_en"],
                "title_gu"        : p["title_gu"],
                "section_heading" : p["section_heading"],
                "passage_index"   : p["passage_index"],
                "passage_gu"      : p["passage_gu"],
            }
            for p in batch_passages
        ]

        collection.add(
            ids=ids,
            embeddings=batch_vectors,  # type: ignore[arg-type]
            documents=documents,
            metadatas=metadatas,       # type: ignore[arg-type]
        )
        print(f"  Inserted {min(i + CHROMA_BATCH, len(all_passages))}/{len(all_passages)}")

    print(f"\n✓ Total passages stored: {collection.count()}\n")

    # 7. Smoke test
    print("─── Smoke test ───────────────────────────────────────────────")
    test_query = "What is the nature of the soul and how should one meditate on God?"
    print(f"Query: \"{test_query}\"")

    q_vec = model.encode(
        [test_query],
        max_length=2048,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"].tolist()

    results = collection.query(
        query_embeddings=q_vec,
        n_results=5,
        include=["metadatas", "distances", "documents"],
    )

    print("\nTop 5 passage results:")
    for i, (meta, dist, doc) in enumerate(zip(
        results["metadatas"][0],   # type: ignore[index]
        results["distances"][0],   # type: ignore[index]
        results["documents"][0],   # type: ignore[index]
    )):
        section = str(meta["section_en"]).replace("_", " ")
        print(f"\n  #{i+1}  Vachanamrut {section} {meta['num_in_section']} "
              f"(#{meta['vachno']}) — dist: {dist:.4f}")
        print(f"       {meta['title_en']}")
        print(f"       {str(doc)[:120]}...")

    print("\n─── Index build complete ─────────────────────────────────────")
    print(f"ChromaDB path  : {CHROMA_PATH}")
    print(f"Collection     : {COLLECTION_NAME}")
    print(f"Passages stored: {collection.count()}")


if __name__ == "__main__":
    main()
