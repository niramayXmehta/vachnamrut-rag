"""
config.py
─────────
Central configuration for the Vachanamrut RAG system.
All other modules import from here — never hardcode paths elsewhere.

LLM mode:
    Set LLM_MODE = "local"  → Ollama / Qwen3:8b  (default, no API key needed)
    Set LLM_MODE = "prod"   → Gemini 2.5 Flash   (requires GEMINI_API_KEY in .env)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

# ─── PROJECT ROOT ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.resolve()

# ─── DATA PATHS ───────────────────────────────────────────────────────────────

MASTER_JSON     = ROOT / "vachanamrut_data" / "vachanamrut_master.json"
SV_MASTER_JSON  = ROOT / "swamini_vaato_data" / "swamini_vaato_master.json"
CHROMA_PATH     = ROOT / "vachanamrut_chroma"
COLLECTION_NAME = "vachanamrut_passages"   # unified collection — both corpora

# ─── EMBEDDING MODEL ──────────────────────────────────────────────────────────

EMBEDDING_MODEL  = "BAAI/bge-m3"
EMBEDDING_DEVICE = "mps"       # M3 Pro — use "cuda" for NVIDIA, "cpu" as fallback
EMBEDDING_FP16   = True        # halves memory usage; negligible quality loss
EMBEDDING_BATCH  = 32          # safe batch size for 18GB RAM

# ─── RETRIEVAL SETTINGS ───────────────────────────────────────────────────────

TOP_K_PASSAGES = 12            # passages sent directly to LLM

# ─── PASSAGE SPLITTING ────────────────────────────────────────────────────────

MIN_PASSAGE_WORDS = 80         # merge paragraph into next if shorter than this

# ─── LLM SETTINGS ─────────────────────────────────────────────────────────────

LLM_MODE = "prod"             # "local" | "prod"

# Local — Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "qwen3:8b"
OLLAMA_THINK    = False        # disable thinking mode for faster responses

# Prod — Gemini
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ─── RELEVANCE THRESHOLD ──────────────────────────────────────────────────────

# Cosine distance above this value → warn user that results may not be relevant
RELEVANCE_THRESHOLD = 0.50

# ─── LANGUAGE DETECTION ───────────────────────────────────────────────────────

# Unicode range for Gujarati script
GUJARATI_UNICODE_START = 0x0A80
GUJARATI_UNICODE_END   = 0x0AFF