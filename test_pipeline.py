"""
test_pipeline.py
────────────────
Terminal test script for the Vachanamrut RAG pipeline.

Usage:
    python test_pipeline.py
    python test_pipeline.py "your question here"
    python test_pipeline.py --stage retrieval
    python test_pipeline.py --stage full
    python test_pipeline.py --stage llm
    python test_pipeline.py --stage raw        ← prints raw LLM output before JSON parsing
    python test_pipeline.py --multiturn
    python test_pipeline.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# ─── DEFAULT QUESTION ─────────────────────────────────────────────────────────

DEFAULT_QUESTION = "What did Swaminarayan say about the importance of ekantik dharma?"

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def hr(char="─", width=65):
    print(char * width)

def section(title: str):
    hr()
    print(f"  {title}")
    hr()

def passage_preview(p: dict, index: int):
    section_str = str(p.get("section_en", "")).replace("_", " ")
    num         = p.get("num_in_section", "")
    vachno      = p.get("vachno", "")
    title       = p.get("title_en", "")
    dist        = p.get("cosine_distance", "?")
    en_text     = p.get("passage_en", "").strip()
    preview     = (en_text[:120] + "...") if len(en_text) > 120 else en_text

    print(f"  #{index}  Vachanamrut {section_str} {num} (#{vachno}) — dist: {dist:.4f}")
    print(f"       {title}")
    print(f"       {preview}")
    print()


# ─── STAGE: RETRIEVAL ─────────────────────────────────────────────────────────

def test_retrieval(question: str) -> list[dict]:
    from backend.retriever import retrieve, is_low_relevance

    section("STAGE 1 — Passage Retrieval (ChromaDB)")
    print(f"  Query      : {question}")

    t0       = time.time()
    passages = retrieve(question)
    elapsed  = time.time() - t0

    low = is_low_relevance(passages)
    print(f"  Retrieved {len(passages)} passages in {elapsed:.2f}s")
    print(f"  Low relevance: {low}")
    print()

    for i, p in enumerate(passages, 1):
        passage_preview(p, i)

    return passages


# ─── STAGE: LLM ───────────────────────────────────────────────────────────────

def test_llm(question: str, passages: list[dict]) -> dict:
    from backend.llm import generate_answer

    section("STAGE 2 — LLM Answer Generation")
    print(f"  Passages supplied : {len(passages)}")
    print()

    t0           = time.time()
    llm_response = generate_answer(question, passages)
    elapsed      = time.time() - t0

    print(f"  Generated in {elapsed:.2f}s")
    print()

    answer_en = llm_response.get("answer_en", "")
    answer_gu = llm_response.get("answer_gu", "")

    print("  ── English Answer ──────────────────────────────────────")
    print()
    if answer_en:
        for line in answer_en.split("\n"):
            print(f"  {line}")
    else:
        print("  [EMPTY — JSON parsing likely failed]")
    print()

    if answer_gu:
        print("  ── Gujarati Answer ─────────────────────────────────────")
        print()
        for line in answer_gu.split("\n"):
            print(f"  {line}")
        print()

    return llm_response


# ─── STAGE: FULL PIPELINE ─────────────────────────────────────────────────────

def test_full(question: str, print_json: bool = False) -> dict:
    from backend.pipeline import query

    print("═" * 65)
    print("  Vachanamrut RAG — Full Pipeline Test")
    print("═" * 65)
    print(f"  Question : {question}")
    hr()

    t0      = time.time()
    result  = query(question)
    elapsed = time.time() - t0

    print(f"\n  Total time   : {elapsed:.2f}s")
    print(f"  Language     : {result['query_lang']}")
    print(f"  Low relevance: {result['low_relevance']}")
    print(f"  Passages used: {len(result['passages'])}")
    print()

    hr()
    print("  ENGLISH ANSWER")
    hr()
    answer_en = result.get("answer_en", "")
    if answer_en:
        for line in answer_en.split("\n"):
            print(f"  {line}")
    else:
        print("  [EMPTY — check LLM response parsing]")
    print()

    answer_gu = result.get("answer_gu", "")
    if answer_gu:
        hr()
        print("  GUJARATI ANSWER")
        hr()
        for line in answer_gu.split("\n"):
            print(f"  {line}")
        print()

    hr()
    print("  CITATIONS")
    hr()
    for c in result.get("citations", []):
        section_str = str(c.get("section_en", "")).replace("_", " ")
        print(f"  · {section_str} {c['num_in_section']} — {c['title_en']}")
    print()

    if print_json:
        hr("═")
        print("  RAW JSON RESULT")
        hr("═")
        clean = {k: v for k, v in result.items() if k != "passages"}
        print(json.dumps(clean, ensure_ascii=False, indent=2))

    return result


# ─── STAGE: MULTITURN ─────────────────────────────────────────────────────────

def test_multiturn():
    from backend.pipeline import query

    print("═" * 65)
    print("  Vachanamrut RAG — Multi-turn Test")
    print("═" * 65)

    turns = [
        "What is the nature of the atma?",
        "How does that relate to the antahkaran?",
        "And how should one meditate given what you just explained?",
    ]

    history: list[dict] = []

    for i, question in enumerate(turns, 1):
        print(f"\n  Turn {i}: {question}")
        hr()

        result    = query(question, history=history)
        answer_en = result.get("answer_en", "[EMPTY]")
        print(f"  {answer_en[:300]}{'...' if len(answer_en) > 300 else ''}")

        history.append({"role": "user",      "content": question})
        history.append({"role": "assistant", "content": answer_en})

    print()
    hr("═")
    print("  Multi-turn test complete.")


# ─── DEBUG: RAW LLM OUTPUT ────────────────────────────────────────────────────

def test_raw_llm(question: str):
    """Print the raw LLM output before JSON parsing — useful for debugging."""
    import requests
    import config
    from backend.retriever import retrieve
    from backend.llm import _build_user_prompt, SYSTEM_PROMPT

    section("DEBUG — Raw LLM Output (before JSON parsing)")
    print(f"  Question: {question}\n")

    passages = retrieve(question)
    prompt   = _build_user_prompt(question, passages)

    print(f"  Sending to Ollama ({config.OLLAMA_MODEL}) ...")
    print()

    payload = {
        "model"  : config.OLLAMA_MODEL,
        "stream" : False,
        "think"  : config.OLLAMA_THINK,
        "options": {"temperature": 0.2, "num_predict": 2048},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }

    r   = requests.post(f"{config.OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
    raw = r.json()["message"]["content"]

    print("  ── Raw output (first 3000 chars) ───────────────────────")
    print(raw[:3000])
    print()
    print(f"  Total length: {len(raw)} chars")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test the Vachanamrut RAG pipeline")
    parser.add_argument(
        "question", nargs="?", default=DEFAULT_QUESTION,
        help="Question to ask (default: hardcoded test question)"
    )
    parser.add_argument(
        "--stage",
        choices=["retrieval", "llm", "full", "raw"],
        default="full",
        help="Which stage to test (default: full)"
    )
    parser.add_argument(
        "--multiturn", action="store_true",
        help="Run a 3-turn conversation test"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print the full JSON result at the end"
    )

    args = parser.parse_args()

    if args.multiturn:
        test_multiturn()
        return

    print("═" * 65)
    print("  Vachanamrut RAG — Pipeline Test")
    print("═" * 65)
    print(f"  Stage    : {args.stage}")
    print(f"  Question : {args.question}")

    if args.stage == "retrieval":
        test_retrieval(args.question)

    elif args.stage == "llm":
        passages = test_retrieval(args.question)
        test_llm(args.question, passages)

    elif args.stage == "raw":
        test_raw_llm(args.question)

    elif args.stage == "full":
        test_full(args.question, print_json=args.json)


if __name__ == "__main__":
    main()
