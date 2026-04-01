"""
backend/pipeline.py
────────────────────
The main RAG pipeline — passage-level ChromaDB retrieval across
Vachanamrut and Swamini Vaato.

Flow:
    1. Detect query language
    2. Embed query (BGE-M3)
    3. Retrieve top-k passages from ChromaDB (optionally filtered by corpus)
    4. Generate answer (Ollama or Gemini)
    5. Return answer + citations + metadata
"""

from __future__ import annotations

import time

import config
from backend.retriever import retrieve, is_low_relevance
from backend.llm       import generate_answer

# ─── LANGUAGE DETECTION ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
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
    Deduplicated citations. Dedup key:
        Vachanamrut   → ("v", vachno)
        Swamini Vaato → ("sv", prakaran, vat_number)
    """
    seen    : set  = set()
    sources : list = []

    for p in passages:
        source = p.get("source", "vachanamrut")

        if source == "vachanamrut":
            key = ("v", p["vachno"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source"          : "vachanamrut",
                "vachno"          : p["vachno"],
                "section_en"      : str(p.get("section_en", "")).replace("_", " "),
                "section_gu"      : p.get("section_gu", ""),
                "num_in_section"  : p.get("num_in_section", ""),
                "title_en"        : p.get("title_en", ""),
                "title_gu"        : p.get("title_gu", ""),
                "section_heading" : p.get("section_heading", ""),
                "cosine_distance" : p.get("cosine_distance", 1.0),
                "prakaran"        : 0,
                "vat_number"      : 0,
            })

        else:  # swamini_vaato
            key = ("sv", p["prakaran"], p["vat_number"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({
                "source"          : "swamini_vaato",
                "prakaran"        : p["prakaran"],
                "vat_number"      : p["vat_number"],
                "cosine_distance" : p.get("cosine_distance", 1.0),
                "vachno"          : 0,
                "section_en"      : "",
                "section_gu"      : "",
                "num_in_section"  : 0,
                "title_en"        : "",
                "title_gu"        : "",
                "section_heading" : "",
            })

    return sources


# ─── HISTORY FORMATTER ────────────────────────────────────────────────────────

def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = []
    for msg in history[-4:]:
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


def _passage_label(p: dict) -> str:
    if p.get("source") == "swamini_vaato":
        return (f"Swamini Vaato  Prakaran {p['prakaran']}  "
                f"Vat {p['vat_number']}  (passage {p['passage_index']})")
    section = str(p.get("section_en", "")).replace("_", " ")
    return (f"Vachanamrut {section} {p.get('num_in_section', '')}  "
            f"(#{p.get('vachno', '')}  passage {p.get('passage_index', 0)})")


def _log_query_trace(
    question    : str,
    query_lang  : str,
    passages    : list[dict],
    citations   : list[dict],
    low_rel     : bool,
    embed_ms    : float,
    retrieve_ms : float,
    llm_ms      : float,
    sources     : list[str],
) -> None:
    total_ms     = embed_ms + retrieve_ms + llm_ms
    divider      = "─" * 72
    corpus_label = " + ".join(sources)

    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  QUERY  [{query_lang.upper()}]  {question[:58]:<58}  ║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"  ⏱  Embed {embed_ms:>6.0f}ms  │  Retrieve {retrieve_ms:>6.0f}ms  │  "
          f"LLM {llm_ms:>6.0f}ms  │  Total {total_ms:>6.0f}ms  │  corpus: {corpus_label}")
    print()
    print(f"  {divider}")
    print(f"  RETRIEVED PASSAGES  (top {len(passages)}, cosine distance — lower = better)")
    print(f"  {divider}")

    for i, p in enumerate(passages, 1):
        dist    = p["cosine_distance"]
        preview = (p.get("passage_en") or p.get("passage_gu") or "").replace("\n", " ").strip()
        preview = (preview[:80] + "…") if len(preview) > 80 else preview
        print()
        print(f"  #{i:>2}  {_passage_label(p)}")
        if p.get("source") == "vachanamrut":
            print(f"       \"{p.get('title_en', '')}\"")
        print(f"       dist={dist:.4f}  {_relevance_label(dist)}")
        print(f"       ↳ {preview}")

    print()
    print(f"  {divider}")
    print(f"  UNIQUE SOURCES CITED ({len(citations)}):")
    for c in citations:
        dist = c.get("cosine_distance", 1.0)
        if c["source"] == "swamini_vaato":
            print(f"    • Swamini Vaato  Prakaran {c['prakaran']}  "
                  f"Vat {c['vat_number']}  dist={dist:.4f}")
        else:
            print(f"    • Vachanamrut {c['section_en']} {c['num_in_section']}"
                  f"  (#{c['vachno']})  dist={dist:.4f}  \"{c.get('title_en', '')}\"")

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
    sources  : list[str] | None = None,
) -> dict:
    """
    Run the full RAG pipeline for a user question.

    Args:
        question : the user's question (Gujarati or English)
        history  : list of {"role": ..., "content": ...} dicts for this session
        sources  : which corpora to search — ["vachanamrut", "swamini_vaato"]
                   None means search both (default)
    """
    history = history or []
    sources = sources or ["vachanamrut", "swamini_vaato"]

    query_lang = detect_language(question)

    t_start        = time.perf_counter()
    passages       = retrieve(question, sources=sources)
    t_retrieve_end = time.perf_counter()

    retrieve_total = (t_retrieve_end - t_start) * 1000
    embed_ms    = retrieve_total * 0.80
    retrieve_ms = retrieve_total * 0.20

    low_relevance = is_low_relevance(passages)

    history_context  = _format_history(history)
    contextual_query = (
        f"{history_context}\nUser: {question}".strip()
        if history_context else question
    )

    t_llm_start  = time.perf_counter()
    llm_response = generate_answer(contextual_query, passages)
    t_llm_end    = time.perf_counter()
    llm_ms       = (t_llm_end - t_llm_start) * 1000

    citations = _build_citations(passages)

    _log_query_trace(
        question    = question,
        query_lang  = query_lang,
        passages    = passages,
        citations   = citations,
        low_rel     = low_relevance,
        embed_ms    = embed_ms,
        retrieve_ms = retrieve_ms,
        llm_ms      = llm_ms,
        sources     = sources,
    )

    return {
        "answer_en"     : llm_response.get("answer_en", ""),
        "answer_gu"     : llm_response.get("answer_gu", ""),
        "query_lang"    : query_lang,
        "citations"     : citations,
        "low_relevance" : low_relevance,
        "passages"      : passages,
    }