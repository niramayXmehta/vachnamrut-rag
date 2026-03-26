"""
backend/embedder.py
───────────────────
BGE-M3 embedding model — singleton loader and encode helper.

The model is loaded once and reused across all pipeline calls.
Calling get_model() multiple times is safe and cheap after the first load.
"""

from __future__ import annotations

import numpy as np
from FlagEmbedding import BGEM3FlagModel

import config

# ─── SINGLETON ────────────────────────────────────────────────────────────────

_model: BGEM3FlagModel | None = None


def get_model() -> BGEM3FlagModel:
    """
    Load and return the BGE-M3 model.
    Model is initialised once and cached for the lifetime of the process.
    """
    global _model
    if _model is None:
        print(f"[embedder] Loading {config.EMBEDDING_MODEL} on {config.EMBEDDING_DEVICE} ...")
        _model = BGEM3FlagModel(
            config.EMBEDDING_MODEL,
            use_fp16=config.EMBEDDING_FP16,
            device=config.EMBEDDING_DEVICE,
        )
        print("[embedder] Model ready.")
    return _model


# ─── ENCODE ───────────────────────────────────────────────────────────────────

def encode(texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """
    Encode a list of texts into dense BGE-M3 vectors.

    Args:
        texts      : list of strings to embed
        batch_size : override config.EMBEDDING_BATCH if provided

    Returns:
        np.ndarray of shape (len(texts), 1024)
    """
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    model = get_model()
    bs    = batch_size or config.EMBEDDING_BATCH

    all_vecs: list[np.ndarray] = []

    for i in range(0, len(texts), bs):
        batch  = texts[i : i + bs]
        output = model.encode(
            batch,
            batch_size=len(batch),
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        all_vecs.append(output["dense_vecs"])

    return np.vstack(all_vecs).astype(np.float32)


def encode_single(text: str) -> np.ndarray:
    """
    Convenience wrapper — encode a single string.

    Returns:
        np.ndarray of shape (1024,)
    """
    return encode([text])[0]
