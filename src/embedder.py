"""
RAG retrieval layer: semantic search over pre-computed song embeddings.

Workflow:
  1. build_index.py runs once offline → saves embeddings_dev.npy
  2. load_index()   loads songs + embeddings into memory at startup
  3. search()       embeds a natural-language query and returns the k
                    most semantically similar songs
"""

import csv
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_index(csv_path: str, emb_path: str) -> tuple[list[dict], np.ndarray]:
    """Load the song catalog and its pre-computed embeddings from disk."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["id"]            = int(row["id"])
            row["energy"]        = float(row["energy"])
            row["tempo_bpm"]     = float(row["tempo_bpm"])
            row["valence"]       = float(row["valence"])
            row["danceability"]  = float(row["danceability"])
            row["acousticness"]  = float(row["acousticness"])
            songs.append(row)
    embeddings = np.load(emb_path)
    return songs, embeddings


def search(
    query: str,
    songs: list[dict],
    embeddings: np.ndarray,
    k: int = 100,
) -> list[dict]:
    """
    Return the k songs whose embeddings are most similar to the query.

    Uses cosine similarity on L2-normalised vectors (equivalent to dot product
    after normalisation, but avoids scipy dependency).
    """
    model = _get_model()
    query_emb = model.encode([query])

    # L2-normalise both query and catalog embeddings
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    emb_norm   = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    scores  = np.dot(emb_norm, query_norm.T).flatten()
    top_idx = np.argsort(scores)[-k:][::-1]
    return [songs[i] for i in top_idx]
