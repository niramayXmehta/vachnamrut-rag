"""
app.py
──────
Flask web server for the Vachanamrut RAG chatbot.

Serves the frontend and exposes a REST API for the pipeline.

Endpoints:
    GET  /                  — serves the frontend HTML
    POST /api/query         — runs the RAG pipeline, returns answer + citations
    POST /api/session/clear — clears the current session history

Usage:
    python app.py

    # Production (via gunicorn)
    gunicorn app:app --bind 0.0.0.0:5000 --workers 1
    (single worker required — BGE-M3 model is loaded per process)
"""

from __future__ import annotations

import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend import query

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# In-memory session store: { session_id: [{"role": ..., "content": ...}, ...] }
# Sessions are cleared on server restart — by design (per-session memory only)
_sessions: dict[str, list[dict]] = {}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> list[dict]:
    """Return the history list for a session, creating it if it doesn't exist."""
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend."""
    return send_from_directory("frontend", "index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Run the RAG pipeline for a user question.

    Request JSON:
        {
            "question"   : str,          — required
            "session_id" : str           — required (generate on frontend if new)
        }

    Response JSON:
        {
            "answer_en"     : str,
            "answer_gu"     : str,
            "query_lang"    : str,       — "en" | "gu"
            "citations"     : list,
            "low_relevance" : bool,
            "session_id"    : str
        }
    """
    data = request.get_json(silent=True) or {}

    question   = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    # Validate
    if not question:
        return jsonify({"error": "question is required"}), 400
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    # Get session history
    history = _get_session(session_id)

    try:
        result = query(question, history=history)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500

    # Update session history
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": result["answer_en"]})

    return jsonify({
        "answer_en"     : result["answer_en"],
        "answer_gu"     : result["answer_gu"],
        "query_lang"    : result["query_lang"],
        "citations"     : result["citations"],
        "low_relevance" : result["low_relevance"],
        "session_id"    : session_id,
    })


@app.route("/api/session/clear", methods=["POST"])
def api_session_clear():
    """
    Clear the history for a session.

    Request JSON:
        { "session_id": str }

    Response JSON:
        { "cleared": true, "session_id": str }
    """
    data       = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    _sessions.pop(session_id, None)

    return jsonify({"cleared": True, "session_id": session_id})


@app.route("/api/health", methods=["GET"])
def api_health():
    """Simple health check endpoint."""
    import config
    return jsonify({
        "status"   : "ok",
        "llm_mode" : config.LLM_MODE,
    })


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config
    print(f"  LLM mode : {config.LLM_MODE}")
    print(f"  Starting Vachanamrut RAG server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
