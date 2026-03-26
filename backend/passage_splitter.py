"""
backend/passage_splitter.py
────────────────────────────
Option D passage splitting — split a discourse into meaningful passages by:
    1. Splitting on double newlines (natural paragraph boundaries)
    2. Merging any paragraph shorter than MIN_PASSAGE_WORDS into the next one

This respects the natural prose structure of the Vachanamrut without relying
on fragile marker detection. Works for both Gujarati and English text.

Each passage is returned with its source discourse metadata attached,
ready for re-ranking in reranker.py.
"""

from __future__ import annotations

import re

import config

# ─── SPLITTING ────────────────────────────────────────────────────────────────

def _split_paragraphs(text: str) -> list[str]:
    """
    Split text on one or more blank lines.
    Strips leading/trailing whitespace from each paragraph.
    Filters out empty strings.
    """
    paragraphs = re.split(r"\n{2,}", text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_short(paragraphs: list[str], min_words: int) -> list[str]:
    """
    Merge any paragraph shorter than min_words into the following paragraph.
    If the short paragraph is the last one, merge it into the previous one.

    This prevents tiny fragments (e.g. Sanskrit verse lines, section headers)
    from becoming standalone passages with no useful context.
    """
    if not paragraphs:
        return []

    merged: list[str] = []
    buffer = ""

    for para in paragraphs:
        word_count = len(para.split())

        if word_count < min_words:
            # Accumulate into buffer
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                # Prepend buffered short content to this paragraph
                merged.append((buffer + "\n\n" + para).strip())
                buffer = ""
            else:
                merged.append(para)

    # Flush any remaining buffer into the last passage
    if buffer:
        if merged:
            merged[-1] = (merged[-1] + "\n\n" + buffer).strip()
        else:
            merged.append(buffer)

    return merged


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def split_discourse(discourse: dict) -> list[dict]:
    """
    Split a single discourse into passages for both English and Gujarati.

    Each passage dict contains:
        passage_en      : str   — English passage text
        passage_gu      : str   — Gujarati passage text (aligned by paragraph index)
        vachno          : int
        section_en      : str
        section_gu      : str
        num_in_section  : int
        title_en        : str
        title_gu        : str
        section_heading : str
        cosine_distance : float — inherited from coarse retrieval
        passage_index   : int   — position of passage within the discourse

    Note on alignment: English and Gujarati are split independently. Paragraph
    counts may differ slightly. Where indices align they are paired; where they
    don't, the shorter language is padded with an empty string. This is fine
    because re-ranking uses English text and citations show both languages.
    """
    min_words = config.MIN_PASSAGE_WORDS

    body_en = discourse.get("body_en", "").strip()
    body_gu = discourse.get("body_gu", "").strip()

    # Split and merge independently for each language
    paras_en = _merge_short(_split_paragraphs(body_en), min_words) if body_en else []
    paras_gu = _merge_short(_split_paragraphs(body_gu), min_words) if body_gu else []

    # Align by index, padding the shorter list with empty strings
    max_len  = max(len(paras_en), len(paras_gu), 1)
    paras_en = paras_en + [""] * (max_len - len(paras_en))
    paras_gu = paras_gu + [""] * (max_len - len(paras_gu))

    # Build passage dicts
    passages = []
    for idx, (en, gu) in enumerate(zip(paras_en, paras_gu)):
        # Skip passages where both languages are empty
        if not en.strip() and not gu.strip():
            continue

        passages.append({
            "passage_en"      : en,
            "passage_gu"      : gu,
            "vachno"          : discourse["vachno"],
            "section_en"      : discourse["section_en"],
            "section_gu"      : discourse.get("section_gu", ""),
            "num_in_section"  : discourse["num_in_section"],
            "title_en"        : discourse.get("title_en", ""),
            "title_gu"        : discourse.get("title_gu", ""),
            "section_heading" : discourse.get("section_heading", ""),
            "cosine_distance" : discourse.get("cosine_distance", 1.0),
            "passage_index"   : idx,
        })

    return passages


def split_discourses(discourses: list[dict]) -> list[dict]:
    """
    Split a list of discourses into a flat list of passages.
    Convenience wrapper over split_discourse().
    """
    all_passages: list[dict] = []
    for discourse in discourses:
        all_passages.extend(split_discourse(discourse))
    return all_passages
