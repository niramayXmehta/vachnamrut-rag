"""
backend/pipeline.py
────────────────────
The main RAG pipeline — passage-level ChromaDB retrieval across
Vachanamrut and Swamini Vaato.

Search mode flow:
    1. Detect query language
    2. Try direct discourse lookup (section+number or title fuzzy match)
    3. If no match → embed query and retrieve from ChromaDB
    4. Generate answer (Ollama or Gemini)
    5. Return answer + citations + metadata

Continue mode flow:
    1. Detect query language
    2. Skip lookup and ChromaDB entirely
    3. Generate answer from conversation history only
    4. Return answer + citations carried over from previous_passages
"""

from __future__ import annotations

import difflib
import re
import time

import config
from backend.retriever import retrieve, is_low_relevance, get_vachanamrut_data
from backend.llm       import generate_answer, generate_continuation
from backend.passage_splitter import split_discourse


# ─── LANGUAGE DETECTION ───────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    if not text:
        return "en"
    gujarati_chars = sum(
        1 for ch in text
        if config.GUJARATI_UNICODE_START <= ord(ch) <= config.GUJARATI_UNICODE_END
    )
    return "gu" if (gujarati_chars / len(text)) > 0.2 else "en"


# ─── DISCOURSE LOOKUP ─────────────────────────────────────────────────────────

# Maps lowercase aliases → canonical section_en value in master JSON
_SECTION_ALIASES: dict[str, str] = {
    "gp"             : "Gadhada_Pratham",
    "pratham"        : "Gadhada_Pratham",
    "gadhada pratham": "Gadhada_Pratham",
    "gadhada first"  : "Gadhada_Pratham",
    "sarangpur"      : "Sarangpur",
    "sar"            : "Sarangpur",
    "kariyani"       : "Kariyani",
    "kar"            : "Kariyani",
    "loya"           : "Loya",
    "panchala"       : "Panchala",
    "pan"            : "Panchala",
    "gm"             : "Gadhada_Madhya",
    "madhya"         : "Gadhada_Madhya",
    "gadhada madhya" : "Gadhada_Madhya",
    "gadhada middle" : "Gadhada_Madhya",
    "vartal"         : "Vartal",
    "var"            : "Vartal",
    "ahmedabad"      : "Ahmedabad",
    "ahd"            : "Ahmedabad",
    "amd"            : "Ahmedabad",
    "ga"             : "Gadhada_Antya",
    "antya"          : "Gadhada_Antya",
    "gadhada antya"  : "Gadhada_Antya",
    "gadhada last"   : "Gadhada_Antya",
    "supplementary"  : "Supplementary",
    "sup"            : "Supplementary",
}

# Section start vachnos for computing absolute vachno from section+num
_SECTION_STARTS: dict[str, int] = {
    "Gadhada_Pratham": 1,
    "Sarangpur"      : 79,
    "Kariyani"       : 97,
    "Loya"           : 109,
    "Panchala"       : 127,
    "Gadhada_Madhya" : 134,
    "Vartal"         : 201,
    "Ahmedabad"      : 221,
    "Gadhada_Antya"  : 224,
    "Bhugol_Khagol"  : 263,
    "Supplementary"  : 264,
}


def _passages_from_record(record: dict, source: str = "vachanamrut") -> list[dict]:
    """
    Split a master JSON record into passage dicts with cosine_distance=0.0
    (indicating a direct lookup, not a vector search result).
    """
    passages = split_discourse({**record, "cosine_distance": 0.0})
    # Ensure source field is set
    for p in passages:
        p["source"] = source
    return passages


def lookup_discourse(query: str, sources: list[str]) -> list[dict] | None:
    """
    Try to match the query to a specific discourse by:
        1. Section + number pattern (e.g. "Gadhada Pratham 16", "GP 16", "Loya 3")
        2. Swamini Vaato pattern (e.g. "Prakaran 2 Vat 30", "P2 V30")
        3. Title fuzzy match against all Vachanamrut titles

    Returns a list of passage dicts if matched, or None to fall through to
    vector search.

    Only searches corpora included in `sources`.
    """
    q = query.strip().lower()

    # ── Strategy 1: Vachanamrut section + number ──────────────────────────────
    if "vachanamrut" in sources:
        # Try each alias from longest to shortest to avoid partial matches
        # e.g. "gadhada pratham" before "pratham"
        for alias in sorted(_SECTION_ALIASES, key=len, reverse=True):
            # Pattern: <alias> <number>  (with optional punctuation/spaces)
            pattern = re.compile(
                r"\b" + re.escape(alias) + r"[\s\-]*(\d+)\b",
                re.IGNORECASE,
            )
            m = pattern.search(q)
            if m:
                section_en = _SECTION_ALIASES[alias]
                num        = int(m.group(1))
                vachno     = _SECTION_STARTS.get(section_en, 0) + num - 1
                data       = get_vachanamrut_data()
                record     = data.get(str(vachno))
                if record:
                    print(f"[pipeline] Lookup hit: {section_en} {num} → vachno {vachno}")
                    return _passages_from_record(record, "vachanamrut")
                break  # alias matched but number out of range — don't keep trying

        # Also try bare "vachno <N>" or "#<N>"
        m = re.search(r"(?:vachno|vachanamrut\s*#?|discourse\s*#?)\s*(\d+)\b", q)
        if m:
            vachno = int(m.group(1))
            data   = get_vachanamrut_data()
            record = data.get(str(vachno))
            if record:
                print(f"[pipeline] Lookup hit: vachno {vachno}")
                return _passages_from_record(record, "vachanamrut")

    # ── Strategy 2: Swamini Vaato prakaran + vat ──────────────────────────────
    if "swamini_vaato" in sources:
        # Patterns: "Prakaran 2 Vat 30", "P2 V30", "p 2 v 30"
        sv_pattern = re.compile(
            r"p(?:rakaran)?[\s\-]*(\d+)[\s\-,]*v(?:at)?[\s\-]*(\d+)",
            re.IGNORECASE,
        )
        m = sv_pattern.search(q)
        if m:
            from backend.retriever import get_swamini_vaato_data
            prakaran   = int(m.group(1))
            vat_number = int(m.group(2))
            key        = f"{prakaran}_{vat_number}"
            sv_data    = get_swamini_vaato_data()
            record     = sv_data.get(key)
            if record:
                print(f"[pipeline] Lookup hit: Swamini Vaato Prakaran {prakaran} Vat {vat_number}")
                return _passages_from_record(record, "swamini_vaato")

    # ── Strategy 3: Title fuzzy match (Vachanamrut only) ──────────────────────
    if "vachanamrut" in sources:
        data        = get_vachanamrut_data()
        title_map   = {
            r["title_en"].lower(): str(vachno)
            for vachno, r in [(k, v) for k, v in data.items()]
            if r.get("title_en")
        }
        matches = difflib.get_close_matches(q, title_map.keys(), n=1, cutoff=0.55)
        if matches:
            key    = title_map[matches[0]]
            record = data[key]
            print(f"[pipeline] Fuzzy title match: \"{matches[0]}\" → vachno {record['vachno']}")
            return _passages_from_record(record, "vachanamrut")

    return None


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
                "cosine_distance" : p.get("cosine_distance", 0.0),
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
                "cosine_distance" : p.get("cosine_distance", 0.0),
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
    (0.00, "████████  direct   "),   # cosine_distance == 0 → direct lookup
    (0.30, "████████  excellent"),
    (0.40, "██████░░  very good"),
    (0.50, "████░░░░  good     "),
    (0.60, "██░░░░░░  fair     "),
    (1.00, "░░░░░░░░  weak     "),
]

def _relevance_label(dist: float) -> str:
    if dist == 0.0:
        return "████████  direct   "
    for threshold, label in _DIST_THRESHOLDS[1:]:
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


def _log_search_trace(
    question    : str,
    query_lang  : str,
    passages    : list[dict],
    citations   : list[dict],
    low_rel     : bool,
    embed_ms    : float,
    retrieve_ms : float,
    llm_ms      : float,
    sources     : list[str],
    lookup_hit  : bool,
) -> None:
    total_ms     = embed_ms + retrieve_ms + llm_ms
    divider      = "─" * 72
    corpus_label = " + ".join(sources)
    mode_label   = "LOOKUP" if lookup_hit else "VECTOR SEARCH"

    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  SEARCH [{query_lang.upper()}]  {question[:58]:<58}  ║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"  ⏱  Embed {embed_ms:>6.0f}ms  │  Retrieve {retrieve_ms:>6.0f}ms  │  "
          f"LLM {llm_ms:>6.0f}ms  │  Total {total_ms:>6.0f}ms")
    print(f"  📚  corpus: {corpus_label}  │  mode: {mode_label}")
    print()
    print(f"  {divider}")
    print(f"  RETRIEVED PASSAGES  ({len(passages)} passages)")
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
        dist = c.get("cosine_distance", 0.0)
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


def _log_continuation_trace(
    question   : str,
    query_lang : str,
    llm_ms     : float,
    n_citations: int,
) -> None:
    divider = "─" * 72
    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  CONTINUE [{query_lang.upper()}]  {question[:56]:<56}  ║")
    print("╚" + "═" * 70 + "╝")
    print()
    print(f"  ⏱  LLM {llm_ms:>6.0f}ms  │  mode: CONTINUE (no retrieval)")
    print(f"  📎  Citations carried over from previous search ({n_citations} sources)")
    print()
    print(f"  {divider}")
    print()


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def query(
    question          : str,
    history           : list[dict] | None = None,
    sources           : list[str] | None = None,
    mode              : str = "search",
    previous_passages : list[dict] | None = None,
) -> dict:
    """
    Run the RAG pipeline for a user question.

    Args:
        question          : the user's question (Gujarati or English)
        history           : list of {"role": ..., "content": ...} dicts
        sources           : corpora to search — ["vachanamrut", "swamini_vaato"]
                            None means both (default)
        mode              : "search" (default) | "continue"
                            "search"   → lookup + ChromaDB + LLM with passages
                            "continue" → skip retrieval, LLM uses history only
        previous_passages : passages from the last search — required for
                            Continue mode citations, ignored in Search mode
    """
    history = history or []
    sources = sources or ["vachanamrut", "swamini_vaato"]

    query_lang = detect_language(question)

    # ── CONTINUE MODE ─────────────────────────────────────────────────────────
    if mode == "continue":
        t_llm_start  = time.perf_counter()
        llm_response = generate_continuation(question, history)
        llm_ms       = (time.perf_counter() - t_llm_start) * 1000

        # Carry over citations from previous search
        passages  = previous_passages or []
        citations = _build_citations(passages)

        _log_continuation_trace(
            question    = question,
            query_lang  = query_lang,
            llm_ms      = llm_ms,
            n_citations = len(citations),
        )

        return {
            "answer_en"     : llm_response.get("answer_en", ""),
            "answer_gu"     : llm_response.get("answer_gu", ""),
            "query_lang"    : query_lang,
            "citations"     : citations,
            "low_relevance" : False,
            "passages"      : passages,
            "mode"          : "continue",
        }

    # ── SEARCH MODE ───────────────────────────────────────────────────────────

    # Step 1: try direct lookup
    t_start    = time.perf_counter()
    lookup_hit = False
    passages   = lookup_discourse(question, sources)

    if passages is not None:
        lookup_hit  = True
        embed_ms    = 0.0
        retrieve_ms = (time.perf_counter() - t_start) * 1000
    else:
        # Step 2: vector search
        passages       = retrieve(question, sources=sources)
        t_retrieve_end = time.perf_counter()
        retrieve_total = (t_retrieve_end - t_start) * 1000
        embed_ms       = retrieve_total * 0.80
        retrieve_ms    = retrieve_total * 0.20

    low_relevance = False if lookup_hit else is_low_relevance(passages)

    # Build contextual query (prepend last 2 turns of history if present)
    history_context  = _format_history(history)
    contextual_query = (
        f"{history_context}\nUser: {question}".strip()
        if history_context else question
    )

    # Step 3: LLM
    t_llm_start  = time.perf_counter()
    llm_response = generate_answer(contextual_query, passages)
    llm_ms       = (time.perf_counter() - t_llm_start) * 1000

    citations = _build_citations(passages)

    _log_search_trace(
        question    = question,
        query_lang  = query_lang,
        passages    = passages,
        citations   = citations,
        low_rel     = low_relevance,
        embed_ms    = embed_ms,
        retrieve_ms = retrieve_ms,
        llm_ms      = llm_ms,
        sources     = sources,
        lookup_hit  = lookup_hit,
    )

    return {
        "answer_en"     : llm_response.get("answer_en", ""),
        "answer_gu"     : llm_response.get("answer_gu", ""),
        "query_lang"    : query_lang,
        "citations"     : citations,
        "low_relevance" : low_relevance,
        "passages"      : passages,
        "mode"          : "search",
    }