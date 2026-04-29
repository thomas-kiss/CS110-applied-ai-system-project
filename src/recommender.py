import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        # TODO: Implement recommendation logic
        return self.songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        # TODO: Implement explanation logic
        return "Explanation placeholder"


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    Returns a list of song dictionaries.
    """
    print(f"Loading songs from {csv_path}...")
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            songs.append(row)
    return songs


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences. Returns (score, reasons).

    Weights (must sum to 1.0):
        energy         0.28  (was 0.25; absorbed mood's 0.07 proportionally)
        valence        0.23  (was 0.20)
        acousticness   0.22  (was 0.20)
        danceability   0.17  (was 0.15)
        tempo          0.05  (was 0.08; small reduction to keep total = 1.0)
        genre          0.05  (unchanged)
    """
    reasons = []

    # Step 1 — Normalize tempo to 0–1 scale
    song_tempo_norm = (song["tempo_bpm"] - 60) / (160 - 60)
    user_tempo_norm = (user_prefs["preferred_tempo_bpm"] - 60) / (160 - 60)

    # Step 2 — Per-feature similarity
    energy_sim       = 1 - abs(song["energy"]       - user_prefs["preferred_energy"])
    valence_sim      = 1 - abs(song["valence"]      - user_prefs["preferred_valence"])
    acousticness_sim = 1 - abs(song["acousticness"] - user_prefs["preferred_acousticness"])
    danceability_sim = 1 - abs(song["danceability"] - user_prefs["preferred_danceability"])
    tempo_sim        = 1 - abs(song_tempo_norm       - user_tempo_norm)
    genre_score      = 1.0 if song["genre"] == user_prefs["preferred_genre"] else 0.0

    # Step 3 — Weighted sum (no mood term)
    score = (
        energy_sim       * 0.28 +
        valence_sim      * 0.23 +
        acousticness_sim * 0.22 +
        danceability_sim * 0.17 +
        tempo_sim        * 0.05 +
        genre_score      * 0.05
    )

    # Step 4 — Build explanation reasons
    if genre_score == 1.0:
        reasons.append(f"matches your preferred genre ({song['genre']})")
    if energy_sim >= 0.85:
        reasons.append(f"energy is close to your preference ({song['energy']:.2f})")
    if valence_sim >= 0.85:
        reasons.append(f"positivity/valence matches ({song['valence']:.2f})")
    if acousticness_sim >= 0.85:
        reasons.append(f"acousticness matches your taste ({song['acousticness']:.2f})")
    if danceability_sim >= 0.85:
        reasons.append(f"danceability is close to your preference ({song['danceability']:.2f})")
    if tempo_sim >= 0.85:
        reasons.append(f"tempo is close to your preference ({song['tempo_bpm']:.0f} BPM)")

    # Step 5 — Behavioral adjustment (runs last so it doesn't corrupt feature math)
    song_id = song["id"]
    if song_id in user_prefs.get("skipped_ids", []) and song_id in user_prefs.get("liked_ids", []):
        score *= 0.5
        reasons.append("penalized: skipped (overrides prior like)")
    elif song_id in user_prefs.get("skipped_ids", []):
        score *= 0.5
        reasons.append("penalized: you previously skipped this song")
    elif song_id in user_prefs.get("liked_ids", []):
        score = min(score * 1.2, 1.0)
        reasons.append("boosted: you previously liked this song")

    if not reasons:
        reasons.append("partial match across audio features")

    return round(score, 4), reasons


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Score all songs and return the top-k as (song, score, explanation) tuples.
    """
    scored = []
    seen = set()
    for song in songs:
        key = (
            song.get("title", "").lower(),
            song.get("artist", "").lower(),
            song.get("id")
            )     
        if key in seen:
            continue
        seen.add(key)
        s, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons) if reasons else "no strong match"
        scored.append((song, s, explanation))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]