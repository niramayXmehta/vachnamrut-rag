"""
backend/pipeline.py
────────────────────
The main RAG pipeline — simplified to use passage-level ChromaDB retrieval.

Flow:
    1. Detect query language
    2. Embed query (BGE-M3)
    3. Retrieve top-k passages directly from ChromaDB
    4. Generate answer (Ollama or Gemini)
    5. Return answer + citations + metadata

Terminal logging prints a detailed trace for every query so you can see
exactly which passages were retrieved and how relevant they are.
"""

from __future__ import annotations

import time

import config
from backend.retriever import retrieve, is_low_relevance
from backend.llm       import generate_answer

# ─── LANGUAGE DETECTION ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Detect whether the query is primarily Gujarati or English.
    Returns "gu" if >20% of characters are Gujarati Unicode, else "en".
    """
    if not text:
        return "en"
    gujarati_chars = sum(
        1 for ch in text
        if config.GUJARATI_UNICODE_START <= ord(ch) <= config.GUJARATI_UNICODE_END
    )
    return "gu" if (gujarati_chars / len(text)) > 0.2 else "en"


# ─── CITATION BUILDER ─────────────────────────────────────────────────────────

def _build_citations(passages: list[dict]) -> list[dict]:
    """
    Deduplicated list of source citations from the retrieved passages.
    Multiple passages from the same discourse collapse into one citation.
    """
    seen    : set  = set()
    sources : list = []

    for p in passages:
        vachno = p["vachno"]
        if vachno in seen:
            continue
        seen.add(vachno)
        section = str(p.get("section_en", "")).replace("_", " ")
        sources.append({
            "vachno"          : vachno,
            "section_en"      : section,
            "section_gu"      : p.get("section_gu", ""),
            "num_in_section"  : p.get("num_in_section", ""),
            "title_en"        : p.get("title_en", ""),
            "title_gu"        : p.get("title_gu", ""),
            "section_heading" : p.get("section_heading", ""),
            "cosine_distance" : p.get("cosine_distance", 1.0),
        })

    return sources


# ─── HISTORY FORMATTER ────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    """Format last 4 messages of conversation history for LLM context."""
    if not history:
        return ""
    recent = history[-4:]
    lines  = []
    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ─── TERMINAL LOGGING ─────────────────────────────────────────────────────────

_DIST_THRESHOLDS = [
    (0.30, "████████  excellent"),
    (0.40, "██████░░  very good"),
    (0.50, "████░░░░  good     "),
    (0.60, "██░░░░░░  fair     "),
    (1.00, "░░░░░░░░  weak     "),
]

def _relevance_label(dist: float) -> str:
    for threshold, label in _DIST_THRESHOLDS:
        if dist <= threshold:
            return label
    return "░░░░░░░░  weak     "


def _log_query_trace(
    question    : str,
    query_lang  : str,
    passages    : list[dict],
    citations   : list[dict],
    low_rel     : bool,
    embed_ms    : float,
    retrieve_ms : float,
    llm_ms      : float,
) -> None:
    total_ms = embed_ms + retrieve_ms + llm_ms
    divider  = "─" * 72

    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  QUERY  [{query_lang.upper()}]  {question[:58]:<58}  ║")
    print("╚" + "═" * 70 + "╝")

    # ── Timings ───────────────────────────────────────────────────────────────
    print()
    print(f"  ⏱  Embed {embed_ms:>6.0f}ms  │  "
          f"Retrieve {retrieve_ms:>6.0f}ms  │  "
          f"LLM {llm_ms:>6.0f}ms  │  "
          f"Total {total_ms:>6.0f}ms")

    # ── Retrieved passages ────────────────────────────────────────────────────
    print()
    print(f"  {divider}")
    print(f"  RETRIEVED PASSAGES  (top {len(passages)}, cosine distance — lower = better)")
    print(f"  {divider}")

    # Group passages by discourse for a cleaner view
    seen_vachno: dict[int, int] = {}  # vachno → count of passages from it
    for p in passages:
        v = p["vachno"]
        seen_vachno[v] = seen_vachno.get(v, 0) + 1

    for i, p in enumerate(passages, 1):
        dist     = p["cosine_distance"]
        section  = str(p.get("section_en", "")).replace("_", " ")
        num      = p.get("num_in_section", "")
        vachno   = p["vachno"]
        title_en = p.get("title_en", "")
        p_idx    = p.get("passage_index", 0)
        label    = _relevance_label(dist)

        # Truncate passage preview to one line
        preview_src = p.get("passage_en") or p.get("passage_gu") or ""
        preview     = preview_src.replace("\n", " ").strip()
        preview     = (preview[:80] + "…") if len(preview) > 80 else preview

        print()
        print(f"  #{i:>2}  Vachanamrut {section} {num}  (#{vachno}, passage {p_idx})")
        print(f"       \"{title_en}\"")
        print(f"       dist={dist:.4f}  {label}")
        print(f"       ↳ {preview}")

    # ── Unique discourses cited ───────────────────────────────────────────────
    print()
    print(f"  {divider}")
    unique = list(dict.fromkeys(p["vachno"] for p in passages))
    print(f"  UNIQUE DISCOURSES CITED ({len(unique)}):")
    for c in citations:
        section = c["section_en"]
        n       = c["num_in_section"]
        v       = c["vachno"]
        title   = c.get("title_en", "")
        dist    = c.get("cosine_distance", 1.0)
        count   = seen_vachno.get(v, 1)
        plural  = "passages" if count > 1 else "passage "
        print(f"    • #{v:>3}  {section} {n:<4}  {count} {plural}  "
              f"dist={dist:.4f}  \"{title}\"")

    # ── Relevance warning ─────────────────────────────────────────────────────
    if low_rel:
        print()
        print(f"  ⚠  LOW RELEVANCE — best distance {passages[0]['cosine_distance']:.4f} "
              f"> threshold {config.RELEVANCE_THRESHOLD}")

    print()
    print(f"  {divider}")
    print()


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def query(
    question : str,
    history  : list[dict] | None = None,
) -> dict:
    """
    Run the full RAG pipeline for a user question.

    Args:
        question : the user's question (Gujarati or English)
        history  : list of {"role": ..., "content": ...} dicts for this session

    Returns:
        {
            "answer_en"     : str
            "answer_gu"     : str
            "query_lang"    : str         — "en" | "gu"
            "citations"     : list[dict]
            "low_relevance" : bool
            "passages"      : list[dict]  — the passages sent to LLM
        }
    """
    history = history or []

    # ── Step 1: Detect language ───────────────────────────────────────────────
    query_lang = detect_language(question)

    # ── Step 2 & 3: Embed + retrieve passages directly from ChromaDB ──────────
    t0 = time.perf_counter()
    _   = question  # embed happens inside retrieve(); time the whole call
    t_embed_start = time.perf_counter()
    passages      = retrieve(question)
    t_retrieve_end = time.perf_counter()

    # Approximate split: embedding is roughly the first 80% of retrieve() time
    # (BGE-M3 inference dominates; ChromaDB lookup is fast)
    retrieve_total = (t_retrieve_end - t_embed_start) * 1000
    embed_ms    = retrieve_total * 0.80
    retrieve_ms = retrieve_total * 0.20

    low_relevance = is_low_relevance(passages)

    # ── Step 4: Build contextualised query with history ───────────────────────
    history_context  = _format_history(history)
    contextual_query = (
        f"{history_context}\nUser: {question}".strip()
        if history_context else question
    )

    # ── Step 5: Generate answer ───────────────────────────────────────────────
    t_llm_start  = time.perf_counter()
    llm_response = generate_answer(contextual_query, passages)
    t_llm_end    = time.perf_counter()
    llm_ms       = (t_llm_end - t_llm_start) * 1000

    # ── Step 6: Build citations ───────────────────────────────────────────────
    citations = _build_citations(passages)

    # ── Step 7: Log trace to terminal ────────────────────────────────────────
    _log_query_trace(
        question    = question,
        query_lang  = query_lang,
        passages    = passages,
        citations   = citations,
        low_rel     = low_relevance,
        embed_ms    = embed_ms,
        retrieve_ms = retrieve_ms,
        llm_ms      = llm_ms,
    )

    return {
        "answer_en"     : llm_response.get("answer_en", ""),
        "answer_gu"     : llm_response.get("answer_gu", ""),
        "query_lang"    : query_lang,
        "citations"     : citations,
        "low_relevance" : low_relevance,
        "passages"      : passages,
    }
