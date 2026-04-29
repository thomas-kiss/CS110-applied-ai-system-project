"""
One-time script: encode all songs in kaggle_tracks.csv into embeddings.

Usage:
    python scripts/build_index.py

Reads  data/kaggle_tracks.csv
Writes data/embeddings.npy  (shape: [n_songs, 384])

NOTE: embeddings.npy is ~175MB for the full dataset and is excluded from git.
Run this script once after cloning to generate it locally.
"""

import csv
import sys
import os
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sentence_transformers import SentenceTransformer

CSV_PATH = "data/kaggle_tracks.csv"
EMB_PATH = "data/embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def song_to_text(song: dict) -> str:
    return (
        f"{song['title']} by {song['artist']}. "
        f"Genre: {song['genre']}. Mood: {song['mood']}. "
        f"Energy: {song['energy']}, Valence: {song['valence']}."
    )


def build(csv_path: str, emb_path: str) -> None:
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append(row)

    print(f"Loaded {len(songs)} songs from {csv_path}")
    texts = [song_to_text(s) for s in songs]

    print(f"Encoding with {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    np.save(emb_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} → {emb_path}")


if __name__ == "__main__":
    build(CSV_PATH, EMB_PATH)
