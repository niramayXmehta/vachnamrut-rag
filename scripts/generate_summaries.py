"""
generate_summaries.py
─────────────────────
Phase 2a — Generate structured summaries for all 274 Vachanamrut discourses
using Gemini 2.0 Flash (free tier), then save them back into master JSON.

Usage:
    export GEMINI_API_KEY="your_key_here"
    python generate_summaries.py

Rate limits (free tier):
    - 15 requests per minute  →  script sleeps to stay under this
    - 1,500 requests per day  →  274 discourses is well within limit

Resumable: already-summarised discourses (those with a 'summary' key) are
skipped, so you can safely re-run after a crash.

Output:
    Writes 'summary' field back into each record in vachanamrut_master.json.
"""

import json
import os
import time
import sys
import argparse
from pathlib import Path
import google.generativeai as genai

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MASTER_JSON = Path(__file__).parent.resolve() / "vachanamrut_data" / "vachanamrut_master.json"

MODEL_NAME     = "gemini-2.5-flash-lite"  # free tier, good for short summaries
REQUESTS_PER_MINUTE = 14          # stay just under the 15 rpm limit
SLEEP_BETWEEN  = 60 / REQUESTS_PER_MINUTE   # ~4.3 seconds per request

# ─── PROMPT ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scholarly assistant specialising in Swaminarayan scripture.
You will be given the full text of one Vachanamrut discourse in both Gujarati and English.
Your task is to produce a dense, structured summary for use as a semantic search index entry.

Return ONLY a plain-text summary — no markdown, no bullet points, no headings.
The summary must be a single block of flowing prose, 200–300 words, in English.

The summary must capture:
1. The central spiritual question or topic discussed
2. The key teachings, arguments, or analogies given by Bhagwan Swaminarayan
3. Any named concepts, Sanskrit/Gujarati theological terms (transliterated), or doctrines mentioned
4. The practical guidance or conclusion offered
5. Any notable questioners or dialogue participants named in the text

Be specific. Prefer concrete details over generic phrasing.
Do not start with "This discourse" or "In this Vachanamrut"."""


def build_user_prompt(record: dict) -> str:
    vachno      = record["vachno"]
    section_en  = record["section_en"]
    num         = record["num_in_section"]
    title_en    = record.get("title_en", "")
    title_gu    = record.get("title_gu", "")
    body_en     = record.get("body_en", "").strip()
    body_gu     = record.get("body_gu", "").strip()
    footnotes_en = record.get("footnotes_en", "").strip()

    parts = [
        f"Vachanamrut {section_en.replace('_', ' ')} {num} (discourse #{vachno})",
        f"English title: {title_en}",
        f"Gujarati title: {title_gu}",
        "",
        "=== ENGLISH TEXT ===",
        body_en,
    ]
    if footnotes_en:
        parts += ["", "=== FOOTNOTES (English) ===", footnotes_en]
    parts += ["", "=== GUJARATI TEXT ===", body_gu]

    return "\n".join(parts)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate summary for discourse #1 only, print it, do NOT save to JSON.",
    )
    parser.add_argument(
        "--vachno",
        type=int,
        default=1,
        help="Which discourse to use in --test mode (default: 1).",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )

    # Load master JSON
    print(f"Loading {MASTER_JSON} ...")
    with open(MASTER_JSON, encoding="utf-8") as f:
        data = json.load(f)
    total = len(data)

    # ── TEST MODE ──────────────────────────────────────────────────────────────
    if args.test:
        key    = str(args.vachno)
        record = data.get(key)
        if not record:
            print(f"ERROR: vachno {args.vachno} not found in JSON.")
            sys.exit(1)

        print(f"\n── TEST MODE ── Discourse #{args.vachno} ──────────────────────────")
        print(f"Title (EN): {record.get('title_en', '')}")
        print(f"Title (GU): {record.get('title_gu', '')}")
        print(f"Section   : {record['section_en']} #{record['num_in_section']}")
        print(f"Body EN   : {len(record.get('body_en',''))} chars")
        print(f"Body GU   : {len(record.get('body_gu',''))} chars")
        print("\nCalling Gemini 2.5 Flash Lite ...\n")

        prompt   = build_user_prompt(record)
        response = model.generate_content(prompt)
        summary  = response.text.strip()

        print("─── GENERATED SUMMARY ───────────────────────────────────────────")
        print(summary)
        print(f"\n─── ({len(summary)} chars, ~{len(summary.split())} words) ──────")
        print("\nNothing saved. Run without --test to process all discourses.")
        return
    # ──────────────────────────────────────────────────────────────────────────

    # Count how many already have summaries
    already_done = sum(1 for v in data.values() if v.get("summary", "").strip())
    pending      = total - already_done
    print(f"Total discourses : {total}")
    print(f"Already summarised: {already_done}")
    print(f"Pending          : {pending}")
    if pending == 0:
        print("All summaries already generated. Nothing to do.")
        return

    estimated_minutes = (pending * SLEEP_BETWEEN) / 60
    print(f"Estimated time   : ~{estimated_minutes:.1f} minutes at {REQUESTS_PER_MINUTE} req/min\n")

    # Sort keys numerically so progress is easy to track
    keys = sorted(data.keys(), key=lambda k: int(k))

    done_this_run = 0
    errors        = 0

    for key in keys:
        record = data[key]
        vachno = record["vachno"]

        # Skip already summarised
        if record.get("summary", "").strip():
            continue

        label = f"[{vachno:>3}/274] {record['section_en'].replace('_',' ')} {record['num_in_section']} — {record.get('title_en','')[:50]}"
        print(f"Generating {label} ...", end=" ", flush=True)

        prompt = build_user_prompt(record)

        # Retry up to 3 times on transient failures
        for attempt in range(1, 4):
            try:
                response = model.generate_content(prompt)
                summary  = response.text.strip()

                if len(summary) < 50:
                    raise ValueError(f"Summary suspiciously short ({len(summary)} chars)")

                record["summary"] = summary
                print(f"✓ ({len(summary)} chars)")
                done_this_run += 1
                break

            except Exception as e:
                print(f"✗ attempt {attempt}: {e}")
                if attempt < 3:
                    time.sleep(10 * attempt)   # back off before retry
                else:
                    print(f"  Skipping {vachno} after 3 failures.")
                    record["summary"] = ""     # mark as attempted but failed
                    errors += 1

        # Save after every discourse so progress is never lost
        with open(MASTER_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Rate-limit sleep (skip after last item)
        if done_this_run + errors < pending:
            time.sleep(SLEEP_BETWEEN)

    print(f"\n─── Done ───────────────────────────────")
    print(f"Generated this run : {done_this_run}")
    print(f"Errors / skipped   : {errors}")
    print(f"Total with summaries: {sum(1 for v in data.values() if v.get('summary','').strip())}/{total}")
    if errors:
        print("Re-run the script to retry failed discourses.")


if __name__ == "__main__":
    main()