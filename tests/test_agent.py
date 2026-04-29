"""
Tests for the agentic orchestrator (src/agent.py).

All Gemini calls are mocked — zero API cost.
Songs and embeddings are injected as fixtures to avoid file I/O.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent import run_agent, extract_user_profile


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_song(
    id=1, title="Test Song", artist="Artist", genre="pop",
    mood="happy", energy=0.7, valence=0.7,
    danceability=0.7, acousticness=0.3, tempo_bpm=120.0,
):
    return {
        "id": id, "title": title, "artist": artist,
        "genre": genre, "mood": mood, "energy": energy,
        "valence": valence, "danceability": danceability,
        "acousticness": acousticness, "tempo_bpm": tempo_bpm,
    }


def make_catalog(n=20):
    return [
        make_song(id=i, title=f"Song {i}", artist=f"Artist {i % 5}")
        for i in range(1, n + 1)
    ]


def make_embeddings(n=20, dim=384):
    rng = np.random.default_rng(42)
    emb = rng.random((n, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


SONGS = make_catalog()
EMBEDDINGS = make_embeddings(len(SONGS))


# ── extract_user_profile ──────────────────────────────────────────────────────

def test_extract_profile_mock_returns_required_keys():
    profile = extract_user_profile("upbeat pop", mock=True)
    required = [
        "preferred_genre", "preferred_mood", "preferred_energy",
        "preferred_valence", "preferred_danceability",
        "preferred_acousticness", "preferred_tempo_bpm",
        "liked_ids", "skipped_ids",
    ]
    for key in required:
        assert key in profile, f"Missing key: {key}"


def test_extract_profile_mock_numeric_values_in_range():
    profile = extract_user_profile("anything", mock=True)
    for field in [
        "preferred_energy", "preferred_valence",
        "preferred_danceability", "preferred_acousticness",
    ]:
        assert 0.0 <= profile[field] <= 1.0, f"{field} out of range"


# ── run_agent — valid queries ─────────────────────────────────────────────────

def test_valid_query_returns_five_results():
    result = run_agent(
        "upbeat pop for working out", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert "error" not in result
    assert len(result["results"]) == 5


def test_results_sorted_descending():
    result = run_agent(
        "chill study music", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    scores = [r[1] for r in result["results"]]
    assert scores == sorted(scores, reverse=True)


def test_result_contains_required_keys():
    result = run_agent(
        "sad indie music", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    for key in ("query", "results", "confidence", "flag", "attempts", "profile_used"):
        assert key in result


def test_confidence_is_float_between_zero_and_one():
    result = run_agent(
        "melancholic evening tracks", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert 0.0 <= result["confidence"] <= 1.0


def test_single_attempt_on_high_confidence():
    result = run_agent(
        "happy dance music", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert result["attempts"] == 1


# ── run_agent — invalid queries ───────────────────────────────────────────────

def test_non_music_query_is_rejected(monkeypatch):
    """Simulate guardrail rejecting a non-music query."""
    import agent as agent_module
    monkeypatch.setattr(
        agent_module, "validate_input",
        lambda q, mock=False: (False, "not a music request"),
    )
    result = run_agent(
        "what is the capital of France?",
        mock=True, songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert "error" in result
    assert "Invalid query" in result["error"]


# ── run_agent — retry behaviour ───────────────────────────────────────────────

def test_low_confidence_triggers_retry(monkeypatch):
    """When first validation returns low confidence, agent retries once."""
    import agent as agent_module
    call_count = {"n": 0}

    def mock_validate_output(query, results, mock=False):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (0.4, "try adding energy context")
        return (0.75, "good match")

    monkeypatch.setattr(agent_module, "validate_output", mock_validate_output)
    result = run_agent(
        "vague music request", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert result["attempts"] == 2


def test_max_two_attempts_even_if_confidence_stays_low(monkeypatch):
    """Agent stops after MAX_ATTEMPTS regardless of confidence."""
    import agent as agent_module
    monkeypatch.setattr(
        agent_module, "validate_output",
        lambda q, r, mock=False: (0.3, "still low"),
    )
    result = run_agent(
        "something weird", mock=True,
        songs=SONGS, embeddings=EMBEDDINGS,
    )
    assert result["attempts"] <= 2
