"""
Swamini Vaato Scraper — anirdesh.com
─────────────────────────────────────
Scrapes all 1,514 Vats across 7 Prakarans in Gujarati + English.
Saves a single master JSON: swamini_vaato_data/swamini_vaato_master.json

Usage:
    python scripts/scrape_swamini_vaato.py

Resumable — already-scraped Vats are skipped on re-run.
1.2s delay between requests for polite scraping.
"""

import json
import os
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL   = "https://www.anirdesh.com/vato/index.php"
DELAY      = 1.2          # seconds between requests
INCREMENT  = 25           # Vats per page (max allowed)
OUTPUT_DIR = "swamini_vaato_data"
JSON_PATH  = os.path.join(OUTPUT_DIR, "swamini_vaato_master.json")

# Hard limits per Prakaran (verified manually)
PRAKARAN_COUNTS = {
    1: 372,
    2: 192,
    3: 74,
    4: 140,
    5: 407,
    6: 295,
    7: 34,
}

TOTAL_VATS = sum(PRAKARAN_COUNTS.values())  # 1,514


# ─────────────────────────────────────────────
# FETCH
# ─────────────────────────────────────────────

def fetch_page(prakaran: int, beg: int, lang: str) -> str | None:
    params = {
        "by"        : "prakaran",
        "lang"      : lang,
        "sortby"    : "prakaran",
        "prakaran"  : prakaran,
        "beg"       : beg,
        "increment" : INCREMENT,
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.encoding = "utf-8"
        return r.text
    except Exception as e:
        print(f"  ✗ fetch error (prakaran={prakaran}, beg={beg}, lang={lang}): {e}")
        return None


# ─────────────────────────────────────────────
# PARSE
# ─────────────────────────────────────────────

def parse_vat(soup: BeautifulSoup, vat_idx: int, lang: str) -> tuple[str, str]:
    """
    Extract body text and footnotes for a single Vat from the page soup.

    Args:
        soup    : BeautifulSoup of the full page
        vat_idx : 1-based index of the Vat on this page (matches id suffix)
        lang    : "gu" or "en"

    Returns:
        (body, footnotes) — both str, empty string if not found
    """
    wrap_id   = f"vat_{lang}_{vat_idx}"
    text_cls  = f"text_{lang}"
    wrap_div  = soup.find("div", id=wrap_id)

    if not wrap_div:
        return "", ""

    # ── Body ──────────────────────────────────────────────────────────────────
    vat_div = wrap_div.find("div", class_="vat")
    body    = ""

    if vat_div:
        paragraphs = vat_div.find_all("p", class_=text_cls)
        if paragraphs:
            parts = []
            for p in paragraphs:
                # Remove <sup> footnote markers
                p_copy = BeautifulSoup(str(p), "html.parser").find("p")
                for sup in p_copy.find_all("sup"):
                    sup.decompose()
                text = p_copy.get_text(separator=" ", strip=True)
                if text:
                    parts.append(text)
            body = "\n\n".join(parts)
        # If no text_en/gu paragraphs found → "Translation unavailable" case
        # body stays ""

    # ── Footnotes ─────────────────────────────────────────────────────────────
    footnote_div = wrap_div.find("div", class_="footnote")
    footnotes    = ""

    if footnote_div:
        fn_parts = []
        for p in footnote_div.find_all("p"):
            text = p.get_text(separator=" ", strip=True)
            if text:
                fn_parts.append(text)
        footnotes = "\n\n".join(fn_parts)

    return body, footnotes


def parse_page(html: str, lang: str, beg: int, total_on_page: int) -> list[dict]:
    """
    Parse a full page and return a list of raw Vat dicts for one language.
    Each dict has: vat_number, body, footnotes.
    """
    soup    = BeautifulSoup(html, "html.parser")
    results = []

    for i in range(1, total_on_page + 1):
        vat_number       = beg + i - 1
        body, footnotes  = parse_vat(soup, i, lang)
        results.append({
            "vat_number" : vat_number,
            "body"       : body,
            "footnotes"  : footnotes,
        })

    return results


# ─────────────────────────────────────────────
# PROGRESS
# ─────────────────────────────────────────────

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║      Swamini Vaato Scraper — anirdesh.com            ║")
    print("║      Gujarati + English · 1,514 Vats · 7 Prakarans  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def print_progress(scraped: int, prakaran: int, vat: int, status: str):
    bar_filled = int((scraped / TOTAL_VATS) * 30)
    bar        = "█" * bar_filled + "░" * (30 - bar_filled)
    pct        = (scraped / TOTAL_VATS) * 100
    print(f"  [{bar}] {pct:5.1f}%  |  {scraped:>4}/{TOTAL_VATS}  |  "
          f"P{prakaran} V{vat:<4}  |  {status}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print_banner()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing data if resuming
    master_data: dict[str, dict] = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            master_data = json.load(f)
        print(f"  ↻  Resuming — found {len(master_data)} already scraped.\n")
    else:
        print("  ✦  Starting fresh scrape.\n")

    print(f"  Output : {os.path.abspath(JSON_PATH)}")
    print(f"  Delay  : {DELAY}s between requests")
    print()
    print("─" * 70)

    errors       : list[str] = []
    total_scraped = len(master_data)
    start_time    = time.time()

    for prakaran, vat_count in PRAKARAN_COUNTS.items():

        print(f"\n  ── Prakaran {prakaran}  ({vat_count} Vats) ──────────────────────────────")

        beg = 1
        while beg <= vat_count:
            # How many Vats are on this page?
            page_size = min(INCREMENT, vat_count - beg + 1)

            # Check if all Vats on this page are already scraped
            all_done = all(
                f"{prakaran}_{vat_number}" in master_data
                for vat_number in range(beg, beg + page_size)
            )
            if all_done:
                for vat_number in range(beg, beg + page_size):
                    print_progress(total_scraped, prakaran, vat_number, "✓ already scraped")
                beg += INCREMENT
                continue

            # ── Fetch Gujarati page ───────────────────────────────────────────
            html_gu = fetch_page(prakaran, beg, "gu")
            if not html_gu:
                for vat_number in range(beg, beg + page_size):
                    key = f"{prakaran}_{vat_number}"
                    errors.append(key)
                beg += INCREMENT
                continue
            time.sleep(DELAY)

            # ── Fetch English page ────────────────────────────────────────────
            html_en = fetch_page(prakaran, beg, "en")
            if not html_en:
                for vat_number in range(beg, beg + page_size):
                    key = f"{prakaran}_{vat_number}"
                    errors.append(key)
                beg += INCREMENT
                continue
            time.sleep(DELAY)

            # ── Parse both pages ──────────────────────────────────────────────
            gu_vats = parse_page(html_gu, "gu", beg, page_size)
            en_vats = parse_page(html_en, "en", beg, page_size)

            # ── Build records ─────────────────────────────────────────────────
            for gu, en in zip(gu_vats, en_vats):
                vat_number = gu["vat_number"]
                key        = f"{prakaran}_{vat_number}"

                # Skip if already in master (resumability within a page batch)
                if key in master_data and master_data[key].get("scraped"):
                    print_progress(total_scraped, prakaran, vat_number, "✓ already scraped")
                    continue

                record = {
                    "prakaran"     : prakaran,
                    "vat_number"   : vat_number,
                    "body_gu"      : gu["body"],
                    "body_en"      : en["body"],      # "" if translation unavailable
                    "footnotes_gu" : gu["footnotes"],
                    "footnotes_en" : en["footnotes"],
                    "source"       : "swamini_vaato",
                    "scraped"      : True,
                    "scraped_at"   : datetime.utcnow().isoformat(),
                }

                master_data[key] = record
                total_scraped   += 1

                status = "✓ saved" if en["body"] else "✓ saved (GU only — no EN translation)"
                print_progress(total_scraped, prakaran, vat_number, status)

            # ── Save after every page batch ───────────────────────────────────
            with open(JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(master_data, f, ensure_ascii=False, indent=2)

            beg += INCREMENT

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed    = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    gu_only    = sum(
        1 for v in master_data.values()
        if v.get("scraped") and not v.get("body_en", "").strip()
    )

    print()
    print("─" * 70)
    print()
    print("  ✦  SCRAPE COMPLETE")
    print(f"     Total scraped  : {len(master_data)} Vats")
    print(f"     GU only (no EN): {gu_only} Vats")
    print(f"     Time taken     : {mins}m {secs}s")
    print(f"     Output         : {os.path.abspath(JSON_PATH)}")
    if errors:
        print(f"     ✗ Errors on    : {errors}")
        print(f"       Re-run the script to retry failed Vats.")
    print()


if __name__ == "__main__":
    main()