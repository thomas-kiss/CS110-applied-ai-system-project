# Model Card: VibeMatch AI

## 1. Model Name

**VibeMatch AI — Semantic Music Recommender**
*(Extension of VibeMatch 1.0, CodePath Ai110 Modules 1–3)*

---

## 2. Intended Use

VibeMatch AI is a hybrid music recommender system that generates personalized song recommendations from an 81,000-track Spotify dataset.

It is designed as an applied AI project demonstrating:
- retrieval-augmented semantic search
- deterministic feature-based ranking
- agentic guardrails and retry logic
- natural language to structured preference extraction
- preset-based recommendation flows

It is intended for educational use, portfolio demonstration, and experimentation with recommendation systems.

It is not intended for production deployment or safety-critical use.

---

## 3. How the System Works

When a user interacts with the system, they either select a preset listening profile or describe a music preference in natural language.

### Step 1 — Input Handling
User input is received through CLI (`main.py`):
- preset mode OR
- free-text vibe description

---

### Step 2 — Input Guardrail (Gemini)
A Gemini model checks whether the request is a valid music recommendation query.

- Non-music requests are rejected
- Valid requests continue

---

### Step 3 — Semantic Retrieval (RAG Layer)
For custom text inputs:
- `all-MiniLM-L6-v2` generates embeddings for the query
- cosine similarity is computed against precomputed song embeddings
- top 100 most similar songs are selected from the 81k catalog

For preset profiles:
- semantic retrieval is skipped or minimally used depending on flow
- full candidate set is passed directly into scoring

---

### Step 4 — Profile Construction

Two modes:

#### Preset Mode
Uses hardcoded numeric profiles defined in `main.py`:
- preferred_genre
- preferred_energy
- preferred_valence
- preferred_acousticness
- preferred_danceability
- preferred_tempo_bpm
- liked_ids / skipped_ids

No LLM is used.

#### Custom Mode
Gemini extracts a structured JSON user profile:
- preferred_genre
- preferred_energy
- preferred_valence
- preferred_acousticness
- preferred_danceability
- preferred_tempo_bpm

---

### Step 5 — Genre Filtering
If applicable, songs are filtered by genre match.

Due to dataset inconsistencies (multiple genre spellings), this filter is heuristic and not strictly enforced.

---

### Step 6 — Scoring Engine (Core Logic)

The original `score_song()` function remains the ranking engine.

It computes weighted similarity across features:

- energy: 0.28
- valence: 0.23
- acousticness: 0.22
- danceability: 0.17
- tempo: 0.05
- genre: 0.05

Additional behavioral logic:
- liked songs → score boost (capped at 1.0)
- skipped songs → score penalty (0.5x)
- if both liked and skipped → penalty overrides like

Each score includes explanation strings describing feature matches.

---

### Step 7 — Output Guardrail (Gemini)
After scoring:
- Gemini assigns confidence score (0.0–1.0)
- if confidence < 0.6:
  - system refines query using feedback
  - retries full pipeline (max 2 attempts)

---

### Step 8 — Result Refinement Loop
User may refine results with follow-up prompts such as:
- “more upbeat”
- “less acoustic”
- “faster tempo”

This triggers:
- Gemini profile update
- re-scoring of existing candidate set (no new retrieval)

---

## 4. Data

### Dataset
- 81,343 Spotify tracks (Kaggle dataset)
- 113 genres after deduplication

### Features used:
- energy
- valence
- acousticness
- danceability
- tempo_bpm
- genre

### Mood
Mood labels were removed because they were fully derived from energy and valence and added no additional signal.

---

## 5. AI Components

| Component | Model | Purpose | Mockable |
|----------|------|--------|----------|
| Input Guardrail | Gemini 2.5 Flash | Validate music intent | Yes |
| Profile Extraction | Gemini 2.5 Flash | NL → structured preferences | Yes |
| Output Guardrail | Gemini 2.5 Flash | Score result quality | Yes |
| Semantic Retrieval | all-MiniLM-L6-v2 | Query embedding search | No |
| Scoring Engine | None (Python) | Deterministic ranking | N/A |

All Gemini components support `mock=True` for testing without API calls.

---

## 6. Strengths

- Deterministic and explainable scoring function (`score_song`)
- Hybrid architecture (semantic + rule-based + LLM)
- Preset profiles enable zero-API recommendation mode
- Fully testable without external dependencies
- Guardrail system prevents irrelevant outputs
- Refinement loop improves user control without re-retrieval

---

## 7. Limitations and Bias

- Genre labels are inconsistent in dataset (e.g. rap vs hip-hop)
- Genre weight is low (0.05), so genre can be overridden by audio similarity
- No popularity or cultural relevance signal
- Semantic embeddings may group songs by textual similarity, not sound
- No persistent user memory across sessions
- LLM-generated profiles may occasionally produce imperfect numeric values
- Dataset is static (no live Spotify updates)

---

## 8. Evaluation

### Unit Tests
- 30 tests total
- 0 API calls (fully mocked)
- Covers:
  - score_song correctness
  - recommend_songs ordering
  - behavioral adjustments
  - edge cases (conflicts, caps, penalties)
  - guardrail logic

---

### Evaluation Harness
- 10 predefined queries
- 100% pass rate in mock mode
- average confidence: ~0.85

---

### Observed Behavior

- Preset profiles bypass LLM and run deterministic scoring
- Semantic retrieval improves candidate relevance for natural language input
- Guardrail successfully detects mismatched outputs and triggers retry
- Refinement prompts reliably shift feature distribution

---

## 9. Design Decisions

**Why keep score_song unchanged?**
It ensures reproducibility, testability, and interpretability across all recommendation modes.

---

**Why remove mood?**
Mood was redundant (derived entirely from energy and valence) and over-clustered the dataset.

---

**Why MiniLM embeddings?**
- lightweight
- local execution
- consistent semantic similarity
- no API cost

---

**Why two-stage refinement?**
Avoids expensive re-retrieval while still adapting ranking to updated preferences.

---

**Why guardrail retry loop?**
Improves reliability by detecting low-confidence recommendation sets and re-attempting generation.

---

## 10. Future Work

- integrate live Spotify API for dynamic catalog updates
- normalize genre taxonomy (rap vs hip-hop mapping layer)
- add popularity / trend weighting
- persist user history across sessions
- improve LLM profile validation (schema enforcement)
- train supervised mood classifier instead of heuristics

---

## 11. Reflection

The system demonstrates a layered recommender architecture where:

- semantic retrieval expands candidate coverage
- deterministic scoring ensures stability and explainability
- LLMs act as translators between natural language and structured preference space
- guardrails provide recovery from low-quality outputs

The most important design decision is that the core ranking logic remains deterministic, while AI components enhance interpretation and robustness rather than replace the recommender itself.