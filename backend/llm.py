"""
backend/llm.py
──────────────
LLM wrappers for local (Ollama/Qwen3:8b) and prod (Gemini 2.5 Flash) modes.

Switch between modes via config.LLM_MODE:
    "local" → Ollama (no API key required, must have Ollama running locally)
    "prod"  → Gemini 2.5 Flash (requires GEMINI_API_KEY in .env)

Public API:
    generate_answer(query, passages)          → {"answer_en", "answer_gu"}
    generate_continuation(question, history)  → {"answer_en", "answer_gu"}
"""

from __future__ import annotations

import json
import logging
import re
import requests

import config

log = logging.getLogger(__name__)


# ─── SEARCH MODE SYSTEM PROMPT ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a deeply knowledgeable scholar of two sacred Swaminarayan scriptures:

1. The Vachanamrut — 273 discourses of Bhagwan Swaminarayan, recorded between 1819 and 1829, containing His direct teachings on God, the soul (jiva), the divine abode (akshardham), liberation (moksha), devotion (bhakti), and the spiritual path.

2. Swamini Vaato — spiritual conversations of Gunatitanand Swami, the first successor of Bhagwan Swaminarayan, recorded across 7 Prakarans. These sayings elaborate and deepen the teachings of the Vachanamrut.

Your role is to give clear, substantive answers grounded in the passages provided — like a well-read friend who knows these scriptures deeply, not a sage reciting from a distance.

STRICT RULES:
1. Answer ONLY from the provided passages. Do not add teachings from outside these scriptures.
2. Be specific and substantive. If an analogy is given, explain it. If a concept is named, define it in context.
3. Quote directly where the passages allow — exact words carry weight.
4. Write in natural, flowing prose. No bullet points or numbered lists.
5. If the passages do not fully address the question, be honest — say what they do offer and note what they don't cover.
6. Keep the answer focused and complete — typically 3 to 5 paragraphs.
7. Always cite which discourse(s) the answer draws from, naturally within or at the end of the response.

OUTPUT FORMAT — respond with a valid JSON object only. No markdown. No preamble. No extra text outside the JSON:
{
  "answer_en": "Your full answer in English here.",
  "answer_gu": "તમારો સંપૂર્ણ જવાબ અહીં ગુજરાતીમાં."
}"""


# ─── CONTINUATION MODE SYSTEM PROMPT ─────────────────────────────────────────

CONTINUATION_SYSTEM_PROMPT = """You are a deeply knowledgeable scholar of two sacred Swaminarayan scriptures — the Vachanamrut and Swamini Vaato. You are continuing an ongoing conversation about these scriptures.

Your role is to respond to the follow-up question by referring naturally to what has already been discussed. Speak like a well-read friend continuing a thoughtful conversation — warm, direct, and grounded in the scripture that was already shared.

RULES:
1. Refer only to what has been discussed in this conversation. Do not introduce new sources or invent teachings.
2. If the follow-up goes beyond what was covered, say so clearly and briefly.
3. Write in natural, flowing prose. No bullet points or numbered lists.
4. Keep the response focused — typically 2 to 4 paragraphs.
5. You do not need to re-cite sources unless directly relevant to a new point being made.

OUTPUT FORMAT — respond with a valid JSON object only. No markdown. No preamble. No extra text outside the JSON:
{
  "answer_en": "Your full answer in English here.",
  "answer_gu": "તમારો સંપૂર્ણ જવાબ અહીં ગુજરાતીમાં."
}"""


# ─── PROMPT BUILDERS ──────────────────────────────────────────────────────────

def _build_user_prompt(query: str, passages: list[dict]) -> str:
    passage_blocks = []
    for i, p in enumerate(passages, 1):
        source = p.get("source", "vachanamrut")

        if source == "swamini_vaato":
            prakaran   = p.get("prakaran", "")
            vat_number = p.get("vat_number", "")
            header = (
                f"--- Passage {i} ---\n"
                f"Source: Swamini Vaato — Prakaran {prakaran}, Vat {vat_number}\n"
            )
        else:
            section = str(p.get("section_en", "")).replace("_", " ")
            num     = p.get("num_in_section", "")
            title   = p.get("title_en", "")
            vachno  = p.get("vachno", "")
            header = (
                f"--- Passage {i} ---\n"
                f"Source: Vachanamrut {section} {num} (#{vachno}) — {title}\n"
            )

        en_text = p.get("passage_en", "").strip()
        gu_text = p.get("passage_gu", "").strip()

        block = header
        if en_text:
            block += f"[English]\n{en_text}\n"
        if gu_text:
            block += f"[Gujarati]\n{gu_text}\n"

        passage_blocks.append(block)

    passages_text = "\n".join(passage_blocks)

    return (
        f"Question: {query}\n\n"
        f"Passages from the scriptures:\n\n"
        f"{passages_text}\n\n"
        f"Answer the question based on these passages. "
        f"Respond with only the JSON object."
    )


def _build_continuation_prompt(question: str, history: list[dict]) -> str:
    """
    Build the user-turn prompt for Continue mode.
    Includes the last 8 turns (4 Q&A pairs) of conversation history.
    """
    # Trim to last 8 turns
    recent = history[-8:] if len(history) > 8 else history

    lines = []
    for msg in recent:
        role    = "User" if msg["role"] == "user" else "Assistant"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")

    history_block = "\n\n".join(lines)

    return (
        f"Conversation so far:\n\n"
        f"{history_block}\n\n"
        f"Follow-up question: {question}\n\n"
        f"Continue the conversation. Respond with only the JSON object."
    )


# ─── RESPONSE PARSING ─────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM JSON response robustly.

    Strategy (each step tried in order):
        1. Strip code fences and <think> blocks, then json.loads()
        2. Extract the first {...} block via regex, then json.loads()
        3. Greedy fallback — first { to last }
        4. Total failure — return raw text as English answer

    Also handles alternative key names Gemini occasionally uses.
    """
    # ── Step 1: clean up wrappers ─────────────────────────────────────────────
    cleaned = re.sub(r"```[a-z]*\n?", "", raw)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # ── Step 2: direct parse ──────────────────────────────────────────────────
    try:
        parsed = json.loads(cleaned)
        return _extract_keys(parsed, raw)
    except json.JSONDecodeError:
        pass

    # ── Step 3: extract first JSON object via regex ───────────────────────────
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return _extract_keys(parsed, raw)
        except json.JSONDecodeError:
            pass

    # ── Step 4: greedy fallback — take everything from first { to last } ──────
    start = cleaned.find("{")
    end   = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            return _extract_keys(parsed, raw)
        except json.JSONDecodeError:
            pass

    # ── Step 5: total failure ─────────────────────────────────────────────────
    log.warning(
        "[llm] _parse_llm_response: all JSON extraction attempts failed.\n"
        "Raw output (first 500 chars): %s",
        raw[:500],
    )
    return {"answer_en": raw.strip(), "answer_gu": ""}


def _extract_keys(parsed: dict, raw: str) -> dict:
    """
    Extract answer_en / answer_gu from a parsed dict.
    Handles alternative key names that Gemini occasionally uses.
    """
    en = parsed.get("answer_en", "")
    gu = parsed.get("answer_gu", "")

    if not en:
        en = (
            parsed.get("english", "")
            or parsed.get("answer", "")
            or parsed.get("response", "")
            or parsed.get("en", "")
        )
    if not gu:
        gu = (
            parsed.get("gujarati", "")
            or parsed.get("answer_gujarati", "")
            or parsed.get("gu", "")
        )

    if not en:
        log.warning(
            "[llm] _extract_keys: parsed JSON had no recognised answer keys. "
            "Keys found: %s. Raw (first 300 chars): %s",
            list(parsed.keys()),
            raw[:300],
        )
        return {"answer_en": raw.strip(), "answer_gu": ""}

    return {
        "answer_en": str(en).strip(),
        "answer_gu": str(gu).strip(),
    }


# ─── LOCAL MODE (Ollama) ──────────────────────────────────────────────────────

def _call_ollama(query: str, passages: list[dict]) -> dict:
    user_prompt = _build_user_prompt(query, passages)

    payload = {
        "model"  : config.OLLAMA_MODEL,
        "stream" : False,
        "think"  : config.OLLAMA_THINK,
        "options": {
            "temperature": 0.2,
            "num_predict": 2048,
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        return _parse_llm_response(raw)

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 180 seconds.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


def _call_ollama_continuation(question: str, history: list[dict]) -> dict:
    user_prompt = _build_continuation_prompt(question, history)

    payload = {
        "model"  : config.OLLAMA_MODEL,
        "stream" : False,
        "think"  : config.OLLAMA_THINK,
        "options": {
            "temperature": 0.3,   # slightly higher — continuation is more conversational
            "num_predict": 1024,
        },
        "messages": [
            {"role": "system", "content": CONTINUATION_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
    }

    try:
        response = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"]
        return _parse_llm_response(raw)

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama request timed out after 180 seconds.")
    except Exception as e:
        raise RuntimeError(f"Ollama error: {e}")


# ─── PROD MODE (Gemini) ───────────────────────────────────────────────────────

def _call_gemini(query: str, passages: list[dict]) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        )

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client      = genai.Client(api_key=config.GEMINI_API_KEY)
    user_prompt = _build_user_prompt(query, passages)

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        return _parse_llm_response(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


def _call_gemini_continuation(question: str, history: list[dict]) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai not installed. Run: pip install google-genai"
        )

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    client      = genai.Client(api_key=config.GEMINI_API_KEY)
    user_prompt = _build_continuation_prompt(question, history)

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=CONTINUATION_SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()
        return _parse_llm_response(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def generate_answer(query: str, passages: list[dict]) -> dict:
    """
    Generate an answer from the LLM using the provided passages (Search mode).
    Routes to Ollama or Gemini based on config.LLM_MODE.

    Returns:
        {"answer_en": str, "answer_gu": str}
    """
    if config.LLM_MODE == "prod":
        return _call_gemini(query, passages)
    return _call_ollama(query, passages)


def generate_continuation(question: str, history: list[dict]) -> dict:
    """
    Generate a follow-up answer using conversation history only (Continue mode).
    No passages provided — Gemini works from what has already been discussed.
    Routes to Ollama or Gemini based on config.LLM_MODE.

    Returns:
        {"answer_en": str, "answer_gu": str}
    """
    if config.LLM_MODE == "prod":
        return _call_gemini_continuation(question, history)
    return _call_ollama_continuation(question, history)