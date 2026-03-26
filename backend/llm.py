"""
backend/llm.py
──────────────
LLM wrappers for local (Ollama/Qwen3:8b) and prod (Gemini 2.5 Flash) modes.

Switch between modes via config.LLM_MODE:
    "local" → Ollama (no API key required, must have Ollama running locally)
    "prod"  → Gemini 2.5 Flash (requires GEMINI_API_KEY in .env)
"""

from __future__ import annotations

import json
import logging
import re
import requests

import config

log = logging.getLogger(__name__)

# ─── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a deeply knowledgeable teacher of the Vachanamrut — the 273 sacred discourses of Bhagwan Swaminarayan, recorded between 1819 and 1829. These discourses contain His direct teachings on the nature of God, the soul (jiva), the divine abode (akshardham), liberation (moksha), devotion (bhakti), and the spiritual path.

Your role is to answer the question as a learned, warm teacher would — someone who has studied these scriptures deeply and speaks from genuine understanding, not just recitation.

STRICT RULES:
1. Answer ONLY from the provided passages. Do not add teachings from outside the Vachanamrut.
2. Be specific and substantive. If Swaminarayan gives an analogy, explain it. If He names a concept, define it in context.
3. Quote Swaminarayan directly where the passages allow — His exact words carry weight.
4. Write in natural, flowing prose. Do not use bullet points or numbered lists.
5. If the passages do not fully address the question, be honest — say what the passages do offer, and note what they don't cover.
6. Maintain a tone of reverence and clarity — as if speaking to a sincere seeker.
7. Keep the answer focused and complete — typically 3 to 5 paragraphs.
8. Always cite which Vachanamrut discourse(s) the answer draws from, naturally within or at the end of the response.

OUTPUT FORMAT — respond with a valid JSON object only. No markdown. No preamble. No extra text outside the JSON:
{
  "answer_en": "Your full answer in English here.",
  "answer_gu": "તમારો સંપૂર્ણ જવાબ અહીં ગુજરાતીમાં."
}"""


# ─── PROMPT BUILDER ───────────────────────────────────────────────────────────

def _build_user_prompt(query: str, passages: list[dict]) -> str:
    passage_blocks = []
    for i, p in enumerate(passages, 1):
        section = str(p.get("section_en", "")).replace("_", " ")
        num     = p.get("num_in_section", "")
        title   = p.get("title_en", "")
        vachno  = p.get("vachno", "")
        en_text = p.get("passage_en", "").strip()
        gu_text = p.get("passage_gu", "").strip()

        block = (
            f"--- Passage {i} ---\n"
            f"Source: Vachanamrut {section} {num} (#{vachno}) — {title}\n"
        )
        if en_text:
            block += f"[English]\n{en_text}\n"
        if gu_text:
            block += f"[Gujarati]\n{gu_text}\n"

        passage_blocks.append(block)

    passages_text = "\n".join(passage_blocks)

    return (
        f"Question: {query}\n\n"
        f"Passages from the Vachanamrut:\n\n"
        f"{passages_text}\n\n"
        f"Answer the question based on these passages. "
        f"Write as a knowledgeable teacher explaining to a sincere seeker. "
        f"Respond with only the JSON object."
    )


# ─── RESPONSE PARSING ─────────────────────────────────────────────────────────

def _parse_llm_response(raw: str) -> dict:
    """
    Parse the LLM JSON response robustly.

    Strategy (each step tried in order):
        1. Strip code fences and <think> blocks, then json.loads()
        2. Extract the first {...} block via regex, then json.loads()
        3. Fall back — return raw text as English answer and log a warning

    Also handles alternative key names Gemini occasionally uses
    ("response", "english", "gujarati", etc.).
    """
    # ── Step 1: clean up wrappers ─────────────────────────────────────────────

    # Strip ALL code fence variants: opening and closing, with or without lang tag
    cleaned = re.sub(r"```[a-z]*\n?", "", raw)
    cleaned = cleaned.replace("```", "")

    # Strip <think>...</think> blocks (Qwen3, some Gemini reasoning traces)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)

    cleaned = cleaned.strip()

    # ── Step 2: direct parse ──────────────────────────────────────────────────
    try:
        parsed = json.loads(cleaned)
        return _extract_keys(parsed, raw)
    except json.JSONDecodeError:
        pass

    # ── Step 3: extract first JSON object via regex ───────────────────────────
    # Non-greedy match from first { to the matching } — handles leading prose
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

    # ── Step 5: total failure — log and return raw ────────────────────────────
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
    # Primary keys
    en = parsed.get("answer_en", "")
    gu = parsed.get("answer_gu", "")

    # Fallback key names seen in the wild
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
        # Parsed JSON had no recognisable keys — log and fall back to raw
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
            "temperature"   : 0.2,    # low for factual, grounded answers
            "num_predict"   : 2048,   # enough for 3-5 paragraphs in both languages
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


# ─── PROD MODE (Gemini) ───────────────────────────────────────────────────────

def _call_gemini(query: str, passages: list[dict]) -> dict:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "google-generativeai not installed. Run: pip install google-generativeai"
        )

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    genai.configure(api_key=config.GEMINI_API_KEY)

    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=SYSTEM_PROMPT,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            # Force Gemini to emit valid JSON — eliminates the parsing lottery entirely.
            # Supported on Gemini 1.5 Flash and later.
            response_mime_type="application/json",
        ),
    )

    user_prompt = _build_user_prompt(query, passages)

    try:
        response = model.generate_content(user_prompt)
        raw = response.text.strip()
        return _parse_llm_response(raw)
    except Exception as e:
        raise RuntimeError(f"Gemini error: {e}")


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def generate_answer(query: str, passages: list[dict]) -> dict:
    """
    Generate an answer from the LLM using the provided passages.
    Routes to Ollama or Gemini based on config.LLM_MODE.

    Returns:
        {"answer_en": str, "answer_gu": str}
    """
    if config.LLM_MODE == "prod":
        return _call_gemini(query, passages)
    return _call_ollama(query, passages)