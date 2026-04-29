# VibeMatch AI — Semantic Music Recommender

## Original Project

This project extends **VibeMatch 1.0**, a content-based music recommender built in Modules 1–3 of CodePath Ai110. The original system scored a hand-curated 20-track catalog against a hardcoded user profile using a seven-feature weighted sum (energy, valence, acousticness, danceability, tempo, mood, genre). It demonstrated the fundamentals of feature similarity scoring and behavioral signals (liked/skipped history), but required manually specifying numeric preferences and could only search 20 songs.

---

## What This Project Does

VibeMatch AI extends the original recommender into a full applied AI system. A user describes what they want to listen to in plain English — "relaxing classical orchestral songs" or "aggressive metal for the gym" — and the system returns tailored recommendations from an 81,000-track Spotify catalog. After seeing the initial results, the user can refine with a follow-up prompt ("more upbeat", "less acoustic") and the system re-ranks without re-searching.

The original `score_song()` and `recommend_songs()` functions are preserved unchanged and serve as the reranking engine. All new functionality — semantic search, natural language understanding, guardrails, and agentic retry — wraps around them.

---

## System Architecture

![Architecture Diagram](assets/architecture.png)

### Component Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  INPUT GUARDRAIL  (Gemini)                          │
│  "Is this a music recommendation request?"          │
│  Rejects non-music queries before anything runs.   │
└────────────────────────┬────────────────────────────┘
                         │ valid
                         ▼
┌─────────────────────────────────────────────────────┐
│  RAG RETRIEVAL  (sentence-transformers)             │
│  Encodes query → cosine similarity search           │
│  Returns 100 semantic candidates from 81k catalog   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  PROFILE EXTRACTION  (Gemini, few-shot)             │
│  Natural language → JSON UserProfile                │
│  { preferred_genre, energy, valence, tempo, … }     │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  GENRE FILTER                                       │
│  Narrows candidates to genre matches when possible  │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  SCORING ENGINE  (original score_song())            │
│  Weighted feature sum — unchanged from Module 1     │
│  energy×0.25 + valence×0.20 + acousticness×0.20 +  │
│  danceability×0.15 + tempo×0.08 + mood×0.07 +      │
│  genre×0.05 + behavioral adjustment                 │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  OUTPUT GUARDRAIL  (Gemini)                         │
│  Scores result quality 0.0–1.0                      │
│  confidence < 0.6 → refine query and retry (max 2)  │
└────────────────────────┬────────────────────────────┘
                         │ confidence ≥ 0.6
                         ▼
                   Top 100 results
                         │
                         ▼
              ┌──────────────────┐
              │  User refinement │  "more upbeat" / "less acoustic"
              │  (optional)      │
              └────────┬─────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │  PROFILE REFINEMENT    │
          │  Gemini updates profile│
          │  Re-scores same 100    │
          └────────────┬───────────┘
                       │
                       ▼
                 Final Top 10
```

---

## Setup Instructions

### Prerequisites

- Python 3.10
- A Google Gemini API key (free tier with `gemma-3-27b-it`)
- The Kaggle Spotify Tracks dataset ([download here](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)) — save as `data/dataset.csv`

### 1. Clone and create virtual environment

```bash
git clone https://github.com/your-username/MusicRecommender.git
cd MusicRecommender
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
# GEMINI_API_KEY=your_key_here
```

### 4. Prepare the data catalog (run once)

```bash
python scripts/prepare_data.py --input data/dataset.csv
```

### 5. Build the embedding index (run once — takes 10–20 minutes)

```bash
python scripts/build_index.py
```

This generates `data/embeddings.npy` (~175MB). It is excluded from git and must be built locally.

### 6. Run the AI agent

```bash
python src/main.py --agent
```

### 7. Run the original rule-based recommender

```bash
python src/main.py
```

### 8. Run tests (zero API calls — fully mocked)

```bash
.venv/bin/python -m pytest tests/ -v
```

### 9. Run the evaluation harness

```bash
python tests/test_harness.py
```

---

## Sample Interactions

### Example 1 — Specific genre request

```
Describe what you want to listen to: gangsta rap for a late night drive

Searching catalog...

Top 10 based on "gangsta rap for a late night drive":
Confidence: 0.78  |  Attempts: 1
────────────────────────────────────────
 1. N.Y. State of Mind — Nas
    hip-hop | intense | score: 0.84
 2. C.R.E.A.M. — Wu-Tang Clan
    hip-hop | intense | score: 0.81
 3. Regulate — Warren G
    hip-hop | chill | score: 0.77
...

Refine your results (e.g. 'more upbeat', 'less acoustic', 'faster tempo')
Refinement: make it more aggressive and faster

Refining...

Top 10 refined results:
────────────────────────────────────────
 1. Shook Ones Pt. II — Mobb Deep
    hip-hop | intense | score: 0.88
    matches your preferred genre (hip-hop); energy is close to your preference (0.91)
...
```

### Example 2 — Mood-based request with refinement

```
Describe what you want to listen to: calm acoustic songs for late night studying

Top 10 based on "calm acoustic songs for late night studying":
Confidence: 0.82  |  Attempts: 1
────────────────────────────────────────
 1. Holocene — Bon Iver
    indie | chill | score: 0.89
 2. The Night Will Always Win — Manchester Orchestra
    indie | melancholic | score: 0.85
...

Refinement: something more instrumental, no vocals

Top 10 refined results:
────────────────────────────────────────
 1. Experience — Ludovico Einaudi
    classical | chill | score: 0.91
    acousticness matches your taste (0.92); energy is close to your preference (0.18)
...
```

### Example 3 — Invalid query rejected by guardrail

```
Describe what you want to listen to: write me a Python function

Rejected: This does not appear to be a music recommendation request.
```

---

## Design Decisions

**Why keep the original scoring engine unchanged?**
The weighted feature sum from Module 1 is a well-calibrated reranker. Its weights (energy 25%, valence 20%) reflect Spotify's own published research on which audio features best predict satisfaction. Adding semantic search on top improves candidate quality without replacing logic that was already working.

**Why sentence-transformers instead of a Gemini embedding call?**
`all-MiniLM-L6-v2` runs locally, costs nothing, and produces high-quality semantic embeddings for music descriptions. Using Gemini embeddings would add API cost and latency to every query. Local embeddings are also reproducible — the same query always returns the same candidates.

**Why few-shot prompting for profile extraction?**
The profile extraction prompt includes four worked examples of query → JSON profile mappings. This "fine-tunes" Gemini's behavior for the music domain without any model training. Without the examples, Gemini produces inconsistent JSON structures. With them, field names and value ranges are stable across diverse queries.

**Why a two-stage refinement instead of re-running the full pipeline?**
Re-running the full pipeline on a refinement ("more upbeat") would discard the 100 semantically relevant candidates already retrieved and replace them with a new semantic search. That throws away valid context. Re-scoring the same pool with an updated profile is faster (one Gemini call vs. three), cheaper, and produces more coherent results because the candidate set remains anchored to the original intent.

**Why retry on low confidence instead of just returning results?**
The output guardrail catches cases where the semantic search retrieved musically adjacent but mismatched songs. The retry appends the guardrail's flag message to the query (e.g., "more hip-hop less rock") before re-searching, giving the embedding model better signal. In testing, one retry is usually sufficient to cross the 0.6 confidence threshold.

---

## Testing Summary

**Unit tests — 31 tests, 0 failures, 0 API calls**

All Gemini calls accept `mock=True`, which returns fixed fixtures. Tests cover:

| Suite | Tests | What's covered |
|---|---|---|
| `test_recommender.py` | 13 | scoring math, sort order, behavioral boosts/penalties, conflict resolution |
| `test_agent.py` | 10 | full pipeline, invalid query rejection, low-confidence retry, profile shape |
| `test_guardrails.py` | 8 | input validation pass/fail, output confidence scoring, API error fallback |

**Evaluation harness — 10/10 fixtures passed**

10 predefined queries ran in mock mode. All returned ≥ 5 results with confidence ≥ 0.6. Average confidence: 0.85.

**Live testing observations:**

- Genre-specific queries (hip-hop, metal, jazz) return correctly categorized results after the genre pre-filter was added.
- The confidence guardrail correctly scored 0.20 for metal results returned against a "gangsta rap" query before the genre filter fix — demonstrating the guardrail catches bad outputs rather than silently accepting them.
- Refinement prompts consistently shift audio feature targets in the expected direction (e.g., "more upbeat" raises energy and tempo in the updated profile).

---

## Reflection and Ethics

**Limitations and biases**

The mood labels (happy, intense, chill, melancholic, neutral) are derived from energy and valence thresholds, not real mood annotations. This means 53% of the catalog is labeled "neutral" — a catch-all bucket that reduces the usefulness of mood-based queries. Genre labels from the Kaggle dataset are inconsistent; the same song may appear under "hip-hop" in one version and "rap" in another, which the genre pre-filter cannot reconcile. The system has no concept of song quality or popularity — a well-known classic and an obscure track with identical audio features score identically.

**Misuse potential**

The input guardrail prevents the system from being used as a general-purpose chatbot, but a determined user could frame off-topic requests as music queries. The system has no content moderation beyond genre matching — it cannot filter for explicit content or age-appropriate material. Gemini could produce hallucinated JSON profiles that silently set extreme feature values, though defaults are applied if any field is missing.

**Surprises during reliability testing**

The most surprising result was how effectively the output guardrail detected bad recommendations — scoring 0.20 confidence for metal songs returned against a rap query, well below the 0.6 retry threshold. This happened before the genre filter was added, and the guardrail's flag message ("results do not match hip-hop or rap genre") was used as the retry signal to correct the query.

**Collaboration with AI**

Claude was used throughout this project for architecture planning, code generation, and debugging. One helpful suggestion was the two-stage retrieval design — retrieving 100 semantic candidates first, then re-scoring with the profile rather than running a single-pass search. This mirrors how production recommenders work and significantly improved result quality. One flawed suggestion was the initial monkeypatch strategy in the test suite, which patched `guardrails.validate_input` instead of `agent.validate_input`. Because `agent.py` imports the function directly (`from guardrails import validate_input`), the patch had no effect and tests silently passed for the wrong reason. The fix — patching `agent.validate_input` — only worked correctly once I understood how Python's import system binds names at import time.

---

## File Structure

```
MusicRecommender/
  src/
    recommender.py      — original scoring engine (unchanged from Module 1)
    embedder.py         — RAG layer: load index, semantic search
    agent.py            — agentic orchestrator: profile extraction, retry loop
    guardrails.py       — input/output validation via Gemini
    main.py             — CLI entry points
  scripts/
    prepare_data.py     — one-time: clean Kaggle CSV → kaggle_tracks.csv
    build_index.py      — one-time: encode catalog → embeddings.npy
  data/
    songs.csv           — original 20-track catalog (Module 1, unchanged)
    kaggle_tracks.csv   — 81k-track production catalog (generated, not committed)
    embeddings.npy      — 81k embeddings @ 384 dims (generated, not committed)
  tests/
    test_recommender.py — 13 unit tests for original scoring engine
    test_agent.py       — 10 integration tests for agentic pipeline
    test_guardrails.py  — 8 tests for input/output guardrails
    test_harness.py     — evaluation harness: 10 fixtures, pass/fail report
  assets/
    architecture.png    — system architecture diagram
  requirements.txt
  .env.example
  README.md
  model_card.md
```

---

[**Model Card**](model_card.md)
