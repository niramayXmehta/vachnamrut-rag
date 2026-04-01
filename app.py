"""
app.py
──────
Flask web server for the Vachanamrut RAG chatbot.

Endpoints:
    GET  /                  — serves the frontend HTML
    POST /api/query         — runs the RAG pipeline, returns answer + citations
    POST /api/session/clear — clears the current session history
    GET  /api/health        — health check

Usage:
    python app.py

    # Production (single worker required — BGE-M3 is not fork-safe)
    gunicorn app:app --bind 0.0.0.0:5000 --workers 1
"""

from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend import query

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

_sessions: dict[str, list[dict]] = {}

VALID_SOURCES = {"vachanamrut", "swamini_vaato"}
VALID_MODES = {"search", "continue"}


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _get_session(session_id: str) -> list[dict]:
    if session_id not in _sessions:
        _sessions[session_id] = []
    return _sessions[session_id]


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Run the RAG pipeline for a user question.

    Request JSON:
        {
            "question"          : str,   — required
            "session_id"        : str,   — required
            "sources"           : ["vachanamrut", "swamini_vaato"], — optional
            "mode"              : "search" | "continue",           — optional
            "previous_passages" : list[dict] | None                — optional
        }

    Response JSON:
        {
            "answer_en"     : str,
            "answer_gu"     : str,
            "query_lang"    : str,
            "citations"     : list,
            "low_relevance" : bool,
            "passages"      : list,
            "mode"          : str,
            "session_id"    : str
        }
    """
    data = request.get_json(silent=True) or {}

    question          = (data.get("question") or "").strip()
    session_id        = (data.get("session_id") or "").strip()
    mode              = (data.get("mode") or "search").strip().lower()
    previous_passages = data.get("previous_passages")

    if not question:
        return jsonify({"error": "question is required"}), 400
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if mode not in VALID_MODES:
        return jsonify({"error": "mode must be one of: search, continue"}), 400
    if previous_passages is not None and not isinstance(previous_passages, list):
        return jsonify({"error": "previous_passages must be a list or null"}), 400

    # Parse and validate sources — default to both if not provided
    raw_sources = data.get("sources")
    if raw_sources is None:
        sources = ["vachanamrut", "swamini_vaato"]
    else:
        sources = [s for s in raw_sources if s in VALID_SOURCES]
        if not sources:
            return jsonify({"error": "sources must include at least one of: vachanamrut, swamini_vaato"}), 400

    history = _get_session(session_id)

    try:
        result = query(
            question,
            history=history,
            sources=sources,
            mode=mode,
            previous_passages=previous_passages,
        )
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        app.logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred."}), 500

    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": result["answer_en"]})

    return jsonify({
        "answer_en"     : result["answer_en"],
        "answer_gu"     : result["answer_gu"],
        "query_lang"    : result["query_lang"],
        "citations"     : result["citations"],
        "low_relevance" : result["low_relevance"],
        "passages"      : result.get("passages", []),
        "mode"          : result.get("mode", mode),
        "session_id"    : session_id,
    })


@app.route("/api/session/clear", methods=["POST"])
def api_session_clear():
    data       = request.get_json(silent=True) or {}
    session_id = (data.get("session_id") or "").strip()

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    _sessions.pop(session_id, None)
    return jsonify({"cleared": True, "session_id": session_id})


@app.route("/api/health", methods=["GET"])
def api_health():
    import config
    return jsonify({
        "status"   : "ok",
        "llm_mode" : config.LLM_MODE,
    })


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import config
    print(f"  LLM mode : {config.LLM_MODE}")
    print(f"  Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
