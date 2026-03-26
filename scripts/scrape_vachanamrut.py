"""
Vachanamrut Scraper — anirdesh.com
Scrapes all 274 discourses in Gujarati + English.
Saves as organised folder structure + master JSON.
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_URL   = "https://www.anirdesh.com/vachanamrut/index.php"
TOTAL      = 274
DELAY      = 1.2          # seconds between requests (polite scraping)
OUTPUT_DIR = "vachanamrut_data"
JSON_PATH  = os.path.join(OUTPUT_DIR, "vachanamrut_master.json")

# Section map: vachno range → section name (English + Gujarati)
SECTIONS = [
    (1,   78,  "Gadhada_Pratham",  "ગઢડા પ્રથમ"),
    (79,  96,  "Sarangpur",        "સારંગપુર"),
    (97,  108, "Kariyani",         "કારિયાણી"),
    (109, 126, "Loya",             "લોયા"),
    (127, 133, "Panchala",         "પંચાળા"),
    (134, 200, "Gadhada_Madhya",   "ગઢડા મધ્ય"),
    (201, 220, "Vartal",           "વરતાલ"),
    (221, 223, "Ahmedabad",        "અમદાવાદ"),
    (224, 262, "Gadhada_Antya",    "ગઢડા અંત્ય"),
    (263, 263, "Bhugol_Khagol",    "ભૂગોળ-ખગોળ"),
    (264, 274, "Supplementary",    "વધારાનાં"),
]

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_section(vachno):
    for start, end, name_en, name_gu in SECTIONS:
        if start <= vachno <= end:
            num_in_section = vachno - start + 1
            return name_en, name_gu, num_in_section
    return "Unknown", "અજ્ઞાત", vachno


def fetch_page(vachno, lang="gu"):
    params = {"format": lang, "vachno": vachno}
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        r.encoding = "utf-8"
        return r.text
    except Exception as e:
        return None


def parse_page(html, vachno, lang):
    soup = BeautifulSoup(html, "html.parser")

    # ── Section heading: <h3 class="pra_secno"> ──
    h3 = soup.find("h3", class_="pra_secno")
    section_heading = h3.get_text(strip=True) if h3 else ""

    # ── Title: <h1 class="title_gu"> or <h1 class="title_en"> inside #vach_text ──
    vach_text_div = soup.find("div", id="vach_text")
    title = ""
    if vach_text_div:
        h1 = vach_text_div.find("h1")
        if h1:
            # Remove any <sup> footnote markers from title
            for sup in h1.find_all("sup"):
                sup.decompose()
            title = h1.get_text(strip=True)

    # ── Body: all <p class="text_gu/en"> and <cite class="text_gu/en">
    #    directly inside <div id="vach_text"> ──
    body_paragraphs = []
    if vach_text_div:
        for tag in vach_text_div.find_all(["p", "cite"], recursive=False):
            # Remove <sup> footnote number markers inline
            tag_copy = BeautifulSoup(str(tag), "html.parser").find(tag.name)
            for sup in tag_copy.find_all("sup"):
                sup.decompose()
            text = tag_copy.get_text(separator=" ", strip=True)
            if text:
                body_paragraphs.append(text)

    body = "\n\n".join(body_paragraphs).strip()

    # ── Footnotes: all <p class="text_gu/en"> inside <div id="fn_wrap"> ──
    fn_wrap = soup.find("div", id="fn_wrap")
    footnotes_paragraphs = []
    if fn_wrap:
        for tag in fn_wrap.find_all("p"):
            text = tag.get_text(separator=" ", strip=True)
            if text:
                footnotes_paragraphs.append(text)

    footnotes = "\n\n".join(footnotes_paragraphs).strip()

    return {
        "title": title,
        "section_heading": section_heading,
        "body": body,
        "footnotes": footnotes,
    }


def save_txt(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        Vachanamrut Scraper — anirdesh.com            ║")
    print("║        Gujarati + English · 274 Discourses           ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def print_progress(vachno, section_en, num_in_section, title, status):
    bar_filled = int((vachno / TOTAL) * 30)
    bar = "█" * bar_filled + "░" * (30 - bar_filled)
    pct = (vachno / TOTAL) * 100
    print(f"  [{bar}] {pct:5.1f}%  |  {vachno:>3}/{TOTAL}  |  "
          f"{section_en} {num_in_section}  |  {status}")


# ─────────────────────────────────────────────
# MAIN SCRAPER
# ─────────────────────────────────────────────

def main():
    print_banner()
    print(f"  Output folder : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Delay         : {DELAY}s between requests")
    print(f"  Languages     : Gujarati (gu) + English (en)")
    print()

    # Load existing data if resuming
    master_data = {}
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            master_data = json.load(f)
        print(f"  ↻  Resuming — found {len(master_data)} already scraped.\n")
    else:
        print("  ✦  Starting fresh scrape.\n")

    print("─" * 70)

    errors = []
    start_time = time.time()

    for vachno in range(1, TOTAL + 1):
        key = str(vachno)
        section_en, section_gu, num_in_section = get_section(vachno)

        # Skip if already done
        if key in master_data and master_data[key].get("scraped"):
            print_progress(vachno, section_en, num_in_section,
                           master_data[key].get("title_gu", ""), "✓ already scraped")
            continue

        # ── Fetch Gujarati ──
        html_gu = fetch_page(vachno, "gu")
        if not html_gu:
            print_progress(vachno, section_en, num_in_section, "", "✗ GU fetch failed")
            errors.append(vachno)
            continue
        time.sleep(DELAY)

        # ── Fetch English ──
        html_en = fetch_page(vachno, "en")
        if not html_en:
            print_progress(vachno, section_en, num_in_section, "", "✗ EN fetch failed")
            errors.append(vachno)
            continue
        time.sleep(DELAY)

        # ── Parse ──
        gu = parse_page(html_gu, vachno, "gu")
        en = parse_page(html_en, vachno, "en")

        title_gu = gu["title"] or f"{section_gu} {num_in_section}"
        title_en = en["title"] or f"{section_en} {num_in_section}"

        # ── Build record ──
        record = {
            "vachno":           vachno,
            "section_en":       section_en,
            "section_gu":       section_gu,
            "num_in_section":   num_in_section,
            "title_gu":         title_gu,
            "title_en":         title_en,
            "section_heading":  gu["section_heading"],
            "body_gu":          gu["body"],
            "body_en":          en["body"],
            "footnotes_gu":     gu["footnotes"],
            "footnotes_en":     en["footnotes"],
            "scraped":          True,
            "scraped_at":       datetime.utcnow().isoformat(),
        }

        # ── Save TXT files ──
        # Folder: vachanamrut_data/Gadhada_Pratham/01_Gadhada_Pratham_1/
        num_padded = str(num_in_section).zfill(2)
        folder_name = f"{num_padded}_{section_en}_{num_in_section}"
        folder_path = os.path.join(OUTPUT_DIR, section_en, folder_name)

        # Gujarati text file
        gu_content = f"""વચનામૃત — {section_gu} {num_in_section}
શીર્ષક: {title_gu}
{'═' * 60}

{gu['body']}

{'─' * 60}
પાદટીપો (Footnotes)
{'─' * 60}
{gu['footnotes']}
"""
        save_txt(os.path.join(folder_path, f"gujarati.txt"), gu_content)

        # English text file
        en_content = f"""Vachanamrut — {section_en} {num_in_section}
Title: {title_en}
{'=' * 60}

{en['body']}

{'-' * 60}
Footnotes
{'-' * 60}
{en['footnotes']}
"""
        save_txt(os.path.join(folder_path, f"english.txt"), en_content)

        # Combined file (for embedding both together)
        combined_content = f"""VACHANAMRUT {section_en.upper()} {num_in_section}
Gujarati Title : {title_gu}
English Title  : {title_en}
{'=' * 60}

[GUJARATI]
{gu['body']}

[ENGLISH]
{en['body']}

[FOOTNOTES — GUJARATI]
{gu['footnotes']}
"""
        save_txt(os.path.join(folder_path, f"combined.txt"), combined_content)

        # ── Save to master JSON ──
        master_data[key] = record
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(master_data, f, ensure_ascii=False, indent=2)

        print_progress(vachno, section_en, num_in_section, title_gu, "✓ saved")

    # ── Summary ──
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print()
    print("─" * 70)
    print()
    print("  ✦  SCRAPE COMPLETE")
    print(f"     Total scraped : {len(master_data)} discourses")
    print(f"     Time taken    : {mins}m {secs}s")
    print(f"     Output folder : {os.path.abspath(OUTPUT_DIR)}")
    print(f"     Master JSON   : {os.path.abspath(JSON_PATH)}")
    if errors:
        print(f"     ✗ Errors on   : vachno {errors}")
        print(f"       Re-run the script to retry failed ones.")
    print()
    print("  Folder structure:")
    print(f"  {OUTPUT_DIR}/")
    for _, _, name_en, _ in SECTIONS:
        count = sum(1 for k, v in master_data.items()
                    if v.get("section_en") == name_en)
        if count:
            print(f"  ├── {name_en}/  ({count} discourses)")
    print(f"  └── vachanamrut_master.json")
    print()


if __name__ == "__main__":
    main()
