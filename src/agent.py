"""
Agentic orchestrator: coordinates RAG retrieval, scoring, and Gemini reasoning.

Flow (up to 2 attempts):
  1. validate_input()       — guardrail: reject non-music queries
  2. embedder.search()      — RAG: retrieve 100 semantic candidates
  3. extract_user_profile() — Gemini: parse natural language → UserProfile dict
  4. recommend_songs()      — existing scoring engine: rerank to top-k
  5. validate_output()      — guardrail: score result quality (0–1)
  6. retry if confidence < 0.6, else return results

All Gemini calls accept mock=True so tests run with zero API cost.
"""

import json
import logging
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

from embedder import load_index, search
from recommender import recommend_songs
from guardrails import validate_input, validate_output

logger = logging.getLogger(__name__)

DATA_CSV = "data/kaggle_tracks.csv"
DATA_EMB = "data/embeddings.npy"

CONFIDENCE_THRESHOLD = 0.6
MAX_ATTEMPTS = 2

# ---------------------------------------------------------------------------
# Preset profiles — selected from a menu, no Gemini call needed
# ---------------------------------------------------------------------------
PRESET_PROFILES = {
    "1": {
        "name": "Late Night Drive",
        "description": "Moody, mid-tempo tracks for cruising after dark",
        "preferred_genre": "hip-hop",
        "preferred_energy": 0.60,
        "preferred_valence": 0.35,
        "preferred_danceability": 0.65,
        "preferred_acousticness": 0.20,
        "preferred_tempo_bpm": 95,
        "liked_ids": [],
        "skipped_ids": [],
    },
    "2": {
        "name": "Gym / Workout",
        "description": "High-energy, driving tracks to push through a hard session",
        "preferred_genre": "Techno",
        "preferred_energy": 0.78,
        "preferred_valence": 0.50,
        "preferred_danceability": 0.82,
        "preferred_acousticness": 0.05,
        "preferred_tempo_bpm": 120,
        "liked_ids": [],
        "skipped_ids": [],
    },
    "3": {
        "name": "Focus / Study",
        "description": "Calm, low-distraction background music for deep work",
        "preferred_genre": "classical",
        "preferred_energy": 0.25,
        "preferred_valence": 0.55,
        "preferred_danceability": 0.25,
        "preferred_acousticness": 0.80,
        "preferred_tempo_bpm": 72,
        "liked_ids": [],
        "skipped_ids": [],
    },
    "4": {
        "name": "Party Mode",
        "description": "Upbeat, danceable bangers to keep the energy high",
        "preferred_genre": "dance",
        "preferred_energy": 0.88,
        "preferred_valence": 0.85,
        "preferred_danceability": 0.92,
        "preferred_acousticness": 0.05,
        "preferred_tempo_bpm": 128,
        "liked_ids": [],
        "skipped_ids": [],
    },
    "5": {
        "name": "Rainy Day / Melancholy",
        "description": "Soft, introspective songs for a quiet, reflective mood",
        "preferred_genre": "indie",
        "preferred_energy": 0.30,
        "preferred_valence": 0.25,
        "preferred_danceability": 0.35,
        "preferred_acousticness": 0.70,
        "preferred_tempo_bpm": 80,
        "liked_ids": [],
        "skipped_ids": [],
    },
    "6": {
        "name": "Feel-Good Classics",
        "description": "Warm, familiar tracks with positive energy",
        "preferred_genre": "pop",
        "preferred_energy": 0.72,
        "preferred_valence": 0.80,
        "preferred_danceability": 0.70,
        "preferred_acousticness": 0.25,
        "preferred_tempo_bpm": 115,
        "liked_ids": [],
        "skipped_ids": [],
    },
}

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_PROFILE_PROMPT = """\
Extract a music preference profile from the user's request.
Return JSON only — no explanation outside the JSON.

Examples:

Request: "upbeat pop for working out"
Profile: {{"preferred_genre": "pop", \
"preferred_energy": 0.85, "preferred_valence": 0.75, \
"preferred_danceability": 0.80, "preferred_acousticness": 0.15, \
"preferred_tempo_bpm": 130}}

Request: "calm acoustic songs for studying late at night"
Profile: {{"preferred_genre": "acoustic", \
"preferred_energy": 0.25, "preferred_valence": 0.55, \
"preferred_danceability": 0.35, "preferred_acousticness": 0.85, \
"preferred_tempo_bpm": 75}}

Request: "melancholic indie songs for a rainy day"
Profile: {{"preferred_genre": "indie", \
"preferred_energy": 0.35, "preferred_valence": 0.25, \
"preferred_danceability": 0.40, "preferred_acousticness": 0.60, \
"preferred_tempo_bpm": 90}}

Request: "high-energy dance tracks for a house party"
Profile: {{"preferred_genre": "dance", \
"preferred_energy": 0.90, "preferred_valence": 0.80, \
"preferred_danceability": 0.92, "preferred_acousticness": 0.05, \
"preferred_tempo_bpm": 128}}

Now extract a profile for:
Request: "{query}"
Profile:"""

_MOCK_PROFILE = {
    "preferred_genre": "pop",
    "preferred_energy": 0.70,
    "preferred_valence": 0.70,
    "preferred_danceability": 0.65,
    "preferred_acousticness": 0.30,
    "preferred_tempo_bpm": 120.0,
    "liked_ids": [],
    "skipped_ids": [],
}

MODEL = "gemma-3-27b-it"
_client = None

_REFINE_PROMPT = """\
You have an existing music preference profile:
{profile}

The user wants to refine their search with this additional request:
"{refinement}"

Update the profile values to reflect the refinement. Keep any fields not mentioned unchanged.
Return JSON only — same format as the original profile.
Profile:"""


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def refine_profile(original_profile: dict, refinement: str, mock: bool = False) -> dict:
    """Adjust an existing profile based on a follow-up refinement request."""
    if mock:
        return original_profile.copy()

    prompt = _REFINE_PROMPT.format(
        profile=json.dumps(original_profile, indent=2),
        refinement=refinement,
    )
    try:
        response = _get_client().models.generate_content(
            model=MODEL, contents=prompt,
        )
        updated = _parse_json(response.text)
        for key, val in original_profile.items():
            updated.setdefault(key, val)
        logger.info(
            "Profile refined | genre=%s energy=%.2f",
            updated["preferred_genre"],
            updated["preferred_energy"],
        )
        return updated
    except Exception as exc:
        logger.warning("refine_profile failed (%s) — keeping original", exc)
        return original_profile.copy()


def extract_user_profile(query: str, mock: bool = False) -> dict:
    """Convert a natural-language query into a UserProfile dict."""
    if mock:
        return _MOCK_PROFILE.copy()

    prompt = _PROFILE_PROMPT.format(query=query)
    try:
        response = _get_client().models.generate_content(
            model=MODEL, contents=prompt,
        )
        profile = _parse_json(response.text)

        defaults = {
            "preferred_genre": "pop",
            "preferred_energy": 0.5,
            "preferred_valence": 0.5,
            "preferred_danceability": 0.5,
            "preferred_acousticness": 0.5,
            "preferred_tempo_bpm": 110.0,
            "liked_ids": [],
            "skipped_ids": [],
        }
        for key, val in defaults.items():
            profile.setdefault(key, val)

        logger.info(
            "Profile extracted | genre=%s energy=%.2f",
            profile["preferred_genre"],
            profile["preferred_energy"],
        )
        return profile

    except Exception as exc:
        logger.warning("Profile extraction failed (%s) — using defaults", exc)
        return _MOCK_PROFILE.copy()


def run_agent(
    user_query: str,
    mock: bool = False,
    songs: list | None = None,
    embeddings=None,
    k: int = 5,
    preset_profile: dict | None = None,
) -> dict:
    """
    Run the full recommendation pipeline for a natural-language query.

    Args:
        user_query:     Free-text description of what the user wants.
        mock:           Skip all Gemini calls (for tests).
        songs:          Pre-loaded song list (injected by tests).
        embeddings:     Pre-computed embeddings (injected by tests).
        preset_profile: If provided, skip Gemini profile extraction and use
                        this profile directly (saves API calls for preset flow).

    Returns a dict: query, results, confidence, flag, attempts, profile_used.
    """
    logger.info("=== Agent started | query: '%s' ===", user_query)

    valid, reason = validate_input(user_query, mock=mock)
    if not valid:
        logger.warning("Input rejected: %s", reason)
        return {"error": f"Invalid query: {reason}", "query": user_query}

    if songs is None or embeddings is None:
        songs, embeddings = load_index(DATA_CSV, DATA_EMB)

    profile: dict = {}
    results: list = []
    confidence: float = 0.0
    flag: str = ""
    current_query = user_query

    for attempt in range(1, MAX_ATTEMPTS + 1):
        logger.info("--- Attempt %d ---", attempt)

        candidates = search(current_query, songs, embeddings, k=100)
        logger.info("search_songs: retrieved %d candidates", len(candidates))

        # Use preset profile on first attempt if provided; fall back to
        # extraction on retry so the refined query can update the profile.
        if preset_profile is not None and attempt == 1:
            profile = preset_profile.copy()
            logger.info("Using preset profile: %s", profile.get("name", ""))
        else:
            profile = extract_user_profile(current_query, mock=mock)

        genre = profile.get("preferred_genre", "")
        if genre:
            genre_matches = [
                s for s in candidates
                if s.get("genre", "").lower() == genre.lower()
            ]
            if len(genre_matches) >= 5:
                candidates = genre_matches
            elif genre_matches:
                seen_ids = {id(s) for s in genre_matches}
                rest = [s for s in candidates if id(s) not in seen_ids]
                candidates = genre_matches + rest
            logger.info(
                "genre_filter: genre=%s matched=%d/%d candidates",
                genre, len(genre_matches), len(candidates),
            )

        results = recommend_songs(profile, candidates, k=k)
        top_score = results[0][1] if results else 0.0
        logger.info("score_songs: top score=%.4f", top_score)

        confidence, flag = validate_output(user_query, results, mock=mock)
        logger.info(
            "validate_results: confidence=%.2f | flag='%s'",
            confidence, flag,
        )

        if confidence >= CONFIDENCE_THRESHOLD:
            break

        if attempt < MAX_ATTEMPTS:
            logger.info("Low confidence (%.2f) — refining query", confidence)
            current_query = f"{user_query} {flag}"

    logger.info(
        "=== Agent finished | attempts=%d confidence=%.2f ===",
        attempt, confidence,
    )

    return {
        "query":        user_query,
        "results":      results,
        "candidates":   candidates,
        "confidence":   confidence,
        "flag":         flag,
        "attempts":     attempt,
        "profile_used": profile,
    }