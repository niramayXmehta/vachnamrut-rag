"""
backend/__init__.py
────────────────────
Public interface for the Vachanamrut RAG backend.
app.py and any test scripts only need to import from here.
"""

from backend.pipeline import query, detect_language

__all__ = ["query", "detect_language"]
