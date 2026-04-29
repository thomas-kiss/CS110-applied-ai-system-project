# Model Card: VibeMatch AI

## 1. Model Name

**VibeMatch AI — Semantic Music Recommender**  
*Extension of VibeMatch 1.0 (CodePath Ai110 Modules 1–3)*

---

## 2. Intended Use

VibeMatch AI is a music recommendation system that retrieves and ranks songs from an 81,000-track Spotify dataset based on either a preset listening profile or a natural language user query.

Users can select predefined vibes (e.g., *Workout*, *Focus*, *Late Night Drive*) or describe what they want in plain English (e.g., “calm acoustic songs for studying”). The system then returns ranked song recommendations with explanations.

This project is intended for educational and portfolio use to demonstrate retrieval-augmented generation concepts, feature-based ranking, and natural language interaction with structured recommendation systems.

It is not intended for production deployment and does not include authentication, content moderation guarantees, or persistent user accounts.

---

## 3. How the System Works

When a user interacts with VibeMatch AI, the system follows a multi-stage pipeline:

### 1. Input Handling
The user either selects a preset vibe or enters a natural language description of what they want to listen to.

### 2. Semantic Retrieval (RAG)
The user query (or preset name/description) is encoded using:

- sentence-transformers/all-MiniLM-L6-v2

The embedding is compared against precomputed song embeddings from the full catalog. The top 100 most similar songs are selected as candidates.

Importantly, this retrieval step is driven only by the text query and is independent of the numeric user profile.

### 3. Profile Construction
There are two paths for building the user preference profile:

- Preset profiles: predefined numeric preferences (genre, energy, valence, acousticness, danceability, tempo)
- Natural language profiles: Gemini extracts structured preferences from the user query

The resulting profile is only used in the ranking stage, not retrieval.

### 4. Scoring (Original Engine)
Each candidate song is scored using the original score_song() function from VibeMatch 1.0.

The scoring function computes a weighted similarity between song features and user preferences:

- Energy: 0.28
- Valence: 0.23
- Acousticness: 0.22
- Danceability: 0.17
- Tempo: 0.05
- Genre: 0.05

Behavioral signals are also applied:
- Liked songs receive a score boost
- Skipped songs receive a score penalty

Each result includes human-readable explanations describing why the song was recommended.

### 5. Refinement
Users may optionally refine results with follow-up instructions (e.g., “more upbeat”, “less acoustic”). The system updates the profile and re-ranks the same candidate set without re-running semantic retrieval.

---

## 4. Data

### Dataset
- Source: Kaggle Spotify Tracks Dataset
- Size: ~81,000 songs
- Genres: 100+ genre labels

### Features Used
- energy
- valence
- acousticness
- danceability
- tempo_bpm
- genre

### Preprocessing
- Songs are deduplicated by title and artist
- Audio features are normalized as needed for scoring

### Limitations
- Genre labels are inconsistent across dataset entries
- No explicit mood labels are provided
- Dataset is a static snapshot and does not include new releases

---

## 5. AI Components

| Component | Model | Purpose |
|---|---|---|
| Semantic embeddings | HuggingFace sentence-transformers/all-MiniLM-L6-v2 | Encode songs and queries |
| Profile extraction | Google Gemini (gemma-3-27b-it) | Convert natural language → preferences |
| Input guardrail | Google Gemini (gemma-3-27b-it) | Reject non-music queries |
| Output validation | Google Gemini (gemma-3-27b-it) | Evaluate result quality |

All Gemini-based components support mocked execution for testing without API usage.

---

## 6. Strengths

- Hybrid architecture combining semantic retrieval and feature-based ranking
- Explainable recommendations with feature-level reasoning
- Low-latency preset mode without LLM calls
- Refinement support without re-running retrieval
- Large-scale dataset (81,000 songs)
- Fully testable system with mocked AI components

---

## 7. Limitations and Bias

- No persistent user memory across sessions
- Inconsistent genre labeling in dataset
- No understanding of lyrics, language, or cultural context
- No popularity or trend-based ranking signal
- No explicit mood model
- Embedding retrieval may miss edge-case semantic intent

---

## 8. Evaluation

### Unit Tests
- 31 total tests
- Covers scoring, ranking, behavioral signals, guardrails, and integration logic
- All tests run in mocked mode (no API dependency)

### Evaluation Harness
- 10 predefined query tests
- All passed with correct ranking behavior
- Average system confidence: ~0.85

---

## 9. Future Work

- Integrate live Spotify API for dynamic catalog updates
- Add persistent user profiles (likes/skips over time)
- Improve genre normalization using synonym mapping
- Add popularity and trend-based ranking signals
- Replace embedding model with a music-domain-specific encoder
- Add multilingual query support

---

## 10. Reflection

VibeMatch AI demonstrates a hybrid architecture where semantic retrieval and feature-based ranking are decoupled.

A key design decision was preserving the original score_song() function rather than replacing it. This ensures deterministic, explainable ranking while allowing AI components to focus on interpretation and retrieval.

The final system combines semantic understanding of user intent with structured feature-based ranking, resulting in a system that is both interpretable and scalable.