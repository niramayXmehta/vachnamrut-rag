"""
build_index.py
──────────────
Splits scripture passages, embeds them with BGE-M3, and stores them
in a persistent ChromaDB collection.

Supports two corpora — Vachanamrut and Swamini Vaato — in a single
unified collection. Each passage carries a 'source' metadata field
so the retriever can filter by corpus if needed.

Usage:
    # Add Swamini Vaato to an existing Vachanamrut index (most common)
    python scripts/build_index.py --source swamini_vaato

    # Rebuild only the Vachanamrut index from scratch
    python scripts/build_index.py --source vachanamrut

    # Rebuild everything from scratch (both corpora)
    python scripts/build_index.py --source both

First run downloads BGE-M3 (~570MB) and caches it locally.

ChromaDB document format:
    id         : "v_{vachno}_{passage_idx}"   (Vachanamrut)
                 "sv_{prakaran}_{vat}_{idx}"  (Swamini Vaato)
    embedding  : 1024-dim BGE-M3 dense vector
    document   : primary text (EN if available, else GU)
    metadata   : source, passage_en, passage_gu, + corpus-specific fields
"""

import argparse
import json
import re
import sys
from pathlib import Path

import chromadb
from FlagEmbedding import BGEM3FlagModel

# ─── PATHS ────────────────────────────────────────────────────────────────────

# build_index.py lives in scripts/ — root is one level up
ROOT            = Path(__file__).parent.parent.resolve()
MASTER_JSON     = ROOT / "vachanamrut_data" / "vachanamrut_master.json"
SV_MASTER_JSON  = ROOT / "swamini_vaato_data" / "swamini_vaato_master.json"
CHROMA_PATH     = ROOT / "vachanamrut_chroma"
COLLECTION_NAME = "vachanamrut_passages"
BATCH_SIZE      = 32
MIN_PASSAGE_WORDS = 80


# ─── PASSAGE SPLITTING ────────────────────────────────────────────────────────

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


def _make_passages(body_en: str, body_gu: str) -> tuple[list[str], list[str]]:
    """Split and align English + Gujarati paragraph lists."""
    paras_en = _merge_short(_split_paragraphs(body_en), MIN_PASSAGE_WORDS) if body_en else []
    paras_gu = _merge_short(_split_paragraphs(body_gu), MIN_PASSAGE_WORDS) if body_gu else []
    max_len  = max(len(paras_en), len(paras_gu), 1)
    paras_en = paras_en + [""] * (max_len - len(paras_en))
    paras_gu = paras_gu + [""] * (max_len - len(paras_gu))
    return paras_en, paras_gu


# ─── VACHANAMRUT PASSAGES ─────────────────────────────────────────────────────

def split_vachanamrut(record: dict) -> list[dict]:
    """Split one Vachanamrut discourse into passage dicts."""
    title_en = record.get("title_en", "").strip()
    body_en  = record.get("body_en", "").strip()
    if title_en and body_en:
        body_en = f"Title: {title_en}\n\n{body_en}"

    paras_en, paras_gu = _make_passages(
        body_en,
        record.get("body_gu", "").strip(),
    )
    passages = []
    for idx, (en, gu) in enumerate(zip(paras_en, paras_gu)):
        if not en.strip() and not gu.strip():
            continue
        passages.append({
            "id"              : f"v_{record['vachno']}_{idx}",
            "passage_en"      : en,
            "passage_gu"      : gu,
            "source"          : "vachanamrut",
            "vachno"          : record["vachno"],
            "section_en"      : record["section_en"],
            "section_gu"      : record.get("section_gu", ""),
            "num_in_section"  : record["num_in_section"],
            "title_en"        : record.get("title_en", ""),
            "title_gu"        : record.get("title_gu", ""),
            "section_heading" : record.get("section_heading", ""),
            "passage_index"   : idx,
            # Swamini Vaato fields — empty for Vachanamrut
            "prakaran"        : 0,
            "vat_number"      : 0,
        })
    return passages


# ─── SWAMINI VAATO PASSAGES ───────────────────────────────────────────────────

def split_swamini_vaato(record: dict) -> list[dict]:
    """Split one Swamini Vaato Vat into passage dicts."""
    paras_en, paras_gu = _make_passages(
        record.get("body_en", "").strip(),
        record.get("body_gu", "").strip(),
    )
    prakaran   = record["prakaran"]
    vat_number = record["vat_number"]

    passages = []
    for idx, (en, gu) in enumerate(zip(paras_en, paras_gu)):
        if not en.strip() and not gu.strip():
            continue
        passages.append({
            "id"              : f"sv_{prakaran}_{vat_number}_{idx}",
            "passage_en"      : en,
            "passage_gu"      : gu,
            "source"          : "swamini_vaato",
            "prakaran"        : prakaran,
            "vat_number"      : vat_number,
            "passage_index"   : idx,
            # Vachanamrut fields — empty for Swamini Vaato
            "vachno"          : 0,
            "section_en"      : "",
            "section_gu"      : "",
            "num_in_section"  : 0,
            "title_en"        : "",
            "title_gu"        : "",
            "section_heading" : "",
        })
    return passages


# ─── EMBEDDING ────────────────────────────────────────────────────────────────

def embed_in_batches(model: BGEM3FlagModel, texts: list[str]) -> list[list[float]]:
    all_vectors   = []
    total_batches = -(-len(texts) // BATCH_SIZE)
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{total_batches} "
              f"({len(batch)} passages) ...", end=" ", flush=True)
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=2048,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        all_vectors.extend([v.tolist() for v in output["dense_vecs"]])
        print("✓")
    return all_vectors


# ─── CHROMA HELPERS ───────────────────────────────────────────────────────────

def get_or_create_collection(
    client: chromadb.ClientAPI,
    delete_first: bool,
) -> chromadb.Collection:
    existing = [c.name for c in client.list_collections()]
    if delete_first and COLLECTION_NAME in existing:
        print(f"Deleting existing '{COLLECTION_NAME}' collection ...")
        client.delete_collection(COLLECTION_NAME)
        existing = []

    if COLLECTION_NAME not in existing:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Created collection '{COLLECTION_NAME}'")
    else:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Using existing collection '{COLLECTION_NAME}' "
              f"({collection.count()} passages already indexed)")

    return collection


def insert_passages(
    collection : chromadb.Collection,
    passages   : list[dict],
    vectors    : list[list[float]],
) -> None:
    CHROMA_BATCH = 500
    print(f"Inserting {len(passages)} passages into ChromaDB ...")

    for i in range(0, len(passages), CHROMA_BATCH):
        batch_p = passages[i : i + CHROMA_BATCH]
        batch_v = vectors[i : i + CHROMA_BATCH]

        collection.add(
            ids        = [p["id"] for p in batch_p],
            embeddings = batch_v,           # type: ignore[arg-type]
            documents  = [
                p["passage_en"] if p["passage_en"].strip() else p["passage_gu"]
                for p in batch_p
            ],
            metadatas  = [
                {
                    "source"          : p["source"],
                    "passage_en"      : p["passage_en"],
                    "passage_gu"      : p["passage_gu"],
                    "passage_index"   : p["passage_index"],
                    # Vachanamrut fields
                    "vachno"          : p["vachno"],
                    "section_en"      : p["section_en"],
                    "section_gu"      : p["section_gu"],
                    "num_in_section"  : p["num_in_section"],
                    "title_en"        : p["title_en"],
                    "title_gu"        : p["title_gu"],
                    "section_heading" : p["section_heading"],
                    # Swamini Vaato fields
                    "prakaran"        : p["prakaran"],
                    "vat_number"      : p["vat_number"],
                }
                for p in batch_p
            ],          # type: ignore[arg-type]
        )
        print(f"  Inserted {min(i + CHROMA_BATCH, len(passages))}/{len(passages)}")


# ─── SMOKE TEST ───────────────────────────────────────────────────────────────

def smoke_test(model: BGEM3FlagModel, collection: chromadb.Collection) -> None:
    print("\n─── Smoke test ───────────────────────────────────────────────")
    query = "What is the nature of the soul and how should one meditate on God?"
    print(f"Query: \"{query}\"")

    q_vec = model.encode(
        [query],
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

    print("\nTop 5 results:")
    for i, (meta, dist, doc) in enumerate(zip(
        results["metadatas"][0],   # type: ignore[index]
        results["distances"][0],   # type: ignore[index]
        results["documents"][0],   # type: ignore[index]
    )):
        source = meta.get("source", "unknown")
        if source == "vachanamrut":
            section = str(meta["section_en"]).replace("_", " ")
            label   = f"Vachanamrut {section} {meta['num_in_section']} (#{meta['vachno']})"
        else:
            label   = f"Swamini Vaato Prakaran {meta['prakaran']} Vat {meta['vat_number']}"
        print(f"\n  #{i+1}  {label} — dist: {dist:.4f}")
        print(f"       {str(doc)[:120]}...")

    print("\n─── Smoke test complete ──────────────────────────────────────")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build the ChromaDB passage index.")
    parser.add_argument(
        "--source",
        choices=["vachanamrut", "swamini_vaato", "both"],
        default="both",
        help=(
            "Which corpus to index.\n"
            "  vachanamrut  — rebuild Vachanamrut from scratch (deletes existing)\n"
            "  swamini_vaato — add Swamini Vaato to existing collection\n"
            "  both         — rebuild entire index from scratch (default)"
        ),
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    all_passages: list[dict] = []

    if args.source in ("vachanamrut", "both"):
        print(f"Loading {MASTER_JSON} ...")
        with open(MASTER_JSON, encoding="utf-8") as f:
            vach_data = json.load(f)
        records = sorted(vach_data.values(), key=lambda r: r["vachno"])
        print(f"Loaded {len(records)} Vachanamrut discourses.")

        vach_passages = []
        for record in records:
            vach_passages.extend(split_vachanamrut(record))
        print(f"Split into {len(vach_passages)} passages.\n")
        all_passages.extend(vach_passages)

    if args.source in ("swamini_vaato", "both"):
        print(f"Loading {SV_MASTER_JSON} ...")
        if not SV_MASTER_JSON.exists():
            print(f"ERROR: {SV_MASTER_JSON} not found. Run scrape_swamini_vaato.py first.")
            sys.exit(1)
        with open(SV_MASTER_JSON, encoding="utf-8") as f:
            sv_data = json.load(f)
        sv_records = sorted(
            sv_data.values(),
            key=lambda r: (r["prakaran"], r["vat_number"])
        )
        print(f"Loaded {len(sv_records)} Swamini Vaato Vats.")

        sv_passages = []
        for record in sv_records:
            sv_passages.extend(split_swamini_vaato(record))
        print(f"Split into {len(sv_passages)} passages.\n")
        all_passages.extend(sv_passages)

    print(f"Total passages to embed: {len(all_passages)}\n")

    # ── Load BGE-M3 ───────────────────────────────────────────────────────────
    print("Loading BGE-M3 model ...")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="mps")
    print("Model loaded.\n")

    # ── Embed ─────────────────────────────────────────────────────────────────
    texts = [
        p["passage_en"] if p["passage_en"].strip() else p["passage_gu"]
        for p in all_passages
    ]
    print(f"Embedding {len(texts)} passages in batches of {BATCH_SIZE} ...")
    vectors = embed_in_batches(model, texts)
    print(f"\nEmbedding complete. Vector dim: {len(vectors[0])}\n")

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete and recreate only when rebuilding from scratch
    delete_first = args.source in ("vachanamrut", "both")
    collection   = get_or_create_collection(chroma_client, delete_first)

    # ── Insert ────────────────────────────────────────────────────────────────
    insert_passages(collection, all_passages, vectors)

    print(f"\n✓ Total passages in collection: {collection.count()}\n")

    # ── Smoke test ────────────────────────────────────────────────────────────
    smoke_test(model, collection)

    print(f"\n─── Index build complete ─────────────────────────────────────")
    print(f"ChromaDB path  : {CHROMA_PATH}")
    print(f"Collection     : {COLLECTION_NAME}")
    print(f"Passages stored: {collection.count()}")


if __name__ == "__main__":
    main()
