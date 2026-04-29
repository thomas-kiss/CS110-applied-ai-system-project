# VibeMatch AI — Semantic Music Recommender

## Original Project

This project extends **VibeMatch 1.0**, a content-based music recommender built in Modules 1–3 of CodePath Ai110. The original system scored a hand-curated 20-track catalog against a hardcoded user profile using a six-feature weighted sum (energy, valence, acousticness, danceability, tempo, genre). It demonstrated the fundamentals of feature similarity scoring and behavioral signals (liked/skipped history), but required manually specifying numeric preferences and could only search 20 songs.

---

## What This Project Does

VibeMatch AI extends the original recommender into a full applied AI system. A user selects from a menu of preset listening profiles — or describes what they want in plain English — and the system returns tailored recommendations from an 81,000-track Spotify catalog.

The original `score_song()` and `recommend_songs()` functions are preserved as the reranking engine. All higher-level functionality — presets, semantic retrieval, natural language parsing, and explanation generation — wraps around this deterministic scoring system.

After the initial results, the user can optionally refine their request ("more upbeat", "less acoustic"), and the system re-ranks the same candidate set using updated preferences.

---

## System Architecture

![Architecture Diagram](assets/architecture.png)

### Component Overview

User selects preset OR describes vibe
    │
    ▼
INPUT PARSING
- Preset profile (hardcoded numeric values)
- OR NLP → Gemini extracts UserProfile JSON

    │
    ▼
RAG RETRIEVAL (sentence-transformers)
Encodes query → cosine similarity search
Returns top 100 candidates from 81k catalog

    │
    ▼
SCORING ENGINE (score_song)
Weighted feature similarity:
energy, valence, acousticness, danceability, tempo, genre + behavioral signals

    │
    ▼
RANKING (recommend_songs)
Sorts by score, removes duplicates, returns top K

    │
    ▼
Top 10 results

    │
    ▼
User refinement ("more upbeat", "less acoustic")

    │
    ▼
PROFILE REFINEMENT (optional Gemini step)
Updates preferences → re-scores same candidate set

---

## Setup Instructions

### Install
```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
python src/main.py
```

### Test
```bash
pytest tests/ -v
```

---

## Key Features

### Preset Profiles
Predefined listening modes map directly to numeric feature vectors used by `score_song()`.

### Natural Language Mode
Gemini converts user text into structured preferences:
```json
{
  "preferred_genre": "rap",
  "preferred_energy": 0.9,
  "preferred_valence": 0.6
}
```

### Semantic Retrieval (RAG)
Uses `sentence-transformers/all-MiniLM-L6-v2` to retrieve 100 candidates before scoring.

### Deterministic Scoring Engine
- energy (0.28)
- valence (0.23)
- acousticness (0.22)
- danceability (0.17)
- tempo (0.05)
- genre (0.05)

### Explanation System
Every recommendation includes human-readable reasons:
- energy is close to your preference
- matches your preferred genre
- boosted/skipped behavior signals

---

## Data

- 81,343 Spotify tracks (Kaggle dataset)
- Features: energy, valence, danceability, acousticness, tempo, genre
- No lyrics
- No user persistence

---

## Design Decisions

### Why keep scoring engine?
Deterministic + testable + explainable ranking system.

### Why add RAG?
Improves candidate quality without changing ranking logic.

### Why presets?
Bypass Gemini and ensure stable behavior.

### Why Gemini?
Only used for natural language → structured preferences.

---

## Limitations

- Genre mismatch (rap vs hip-hop)
- No popularity signal
- No cultural awareness
- Static dataset
- No long-term memory

---

## Example Output

Top 10 for "Gym / Workout"

1. Bodywork — Stanton Warriors | score: 0.90  
2. Work Your Body — Fast Eddie | score: 0.89  
3. Body Back — Gryffin | score: 0.89  

---

## Testing

- 30+ unit tests
- Fully mocked (no API calls)
- Covers scoring, ranking, and behavior logic

---

## File Structure

src/
  recommender.py
  main.py
  embedder.py
  agent.py

tests/
  test_recommender.py
  test_agent.py
