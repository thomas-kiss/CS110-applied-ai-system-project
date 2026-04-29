"""
Evaluation harness — stretch feature (+2 points).

Runs the agent against 10 predefined queries in mock mode (zero API calls)
and prints a pass/fail report with confidence scores.

Usage:
    python tests/test_harness.py

A test PASSES when:
  - No error key in the result
  - confidence >= 0.6
  - At least 5 results returned
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import run_agent

PASS_THRESHOLD = 0.6
REQUIRED_RESULTS = 5

# 10 queries covering different vibes, activities, and edge cases
FIXTURES = [
    {"query": "upbeat pop songs for a morning run",         "tags": ["pop", "energetic"]},
    {"query": "late night coding focus beats",              "tags": ["lofi", "chill"]},
    {"query": "sad indie music for a rainy afternoon",      "tags": ["indie", "low-energy"]},
    {"query": "high energy dance tracks for a house party", "tags": ["dance", "energetic"]},
    {"query": "calm acoustic songs for studying",           "tags": ["acoustic", "chill"]},
    {"query": "aggressive metal for the gym",               "tags": ["metal", "high-energy"]},
    {"query": "romantic jazz for a dinner at home",         "tags": ["jazz", "chill"]},
    {"query": "nostalgic 90s vibes driving at night",       "tags": ["pop", "mid-energy"]},
    {"query": "ambient background music for meditation",    "tags": ["ambient", "chill"]},
    {"query": "something angry and fast to wake me up",     "tags": ["rock", "high-energy"]},
]


def make_catalog(n=20):
    return [
        {
            "id": i, "title": f"Song {i}", "artist": f"Artist {i % 5}",
            "genre": "pop", "energy": 0.7, "valence": 0.7,
            "danceability": 0.7, "acousticness": 0.3, "tempo_bpm": 120.0,
        }
        for i in range(1, n + 1)
    ]


def make_embeddings(n=20, dim=384):
    rng = np.random.default_rng(42)
    emb = rng.random((n, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def run_harness() -> None:
    songs = make_catalog(20)
    embeddings = make_embeddings(len(songs))

    passed = 0
    confidences = []

    print("\n" + "=" * 60)
    print("  EVALUATION HARNESS — Music Recommender AI Agent")
    print("=" * 60)

    for i, fixture in enumerate(FIXTURES, 1):
        query = fixture["query"]
        result = run_agent(query, mock=True, songs=songs, embeddings=embeddings)

        has_error       = "error" in result
        enough_results  = not has_error and len(result["results"]) >= REQUIRED_RESULTS
        confidence      = result.get("confidence", 0.0)
        above_threshold = confidence >= PASS_THRESHOLD

        ok = not has_error and enough_results and above_threshold

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        confidences.append(confidence)

        label  = f"[{status}] ({i:02d}) \"{query}\""
        detail = f"confidence: {confidence:.2f}"
        if has_error:
            detail += f" | error: {result['error']}"
        elif not enough_results:
            detail += f" | only {len(result['results'])} results (need {REQUIRED_RESULTS})"
        elif not above_threshold:
            detail += f" | below threshold ({PASS_THRESHOLD})"

        print(f"{label}")
        print(f"         {detail}")

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    total    = len(FIXTURES)

    print("─" * 60)
    print(f"Results        : {passed}/{total} passed")
    print(f"Avg confidence : {avg_conf:.2f}")
    print("=" * 60 + "\n")

    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_harness()