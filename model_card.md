# Model Card: VibeMatch AI

## 1. Model Name

**VibeMatch AI — Semantic Music Recommender**
*(Extended from VibeMatch 1.0, CodePath Ai110 Module 1–3)*

---

## 2. Intended Use

VibeMatch AI recommends songs from an 81,000-track Spotify catalog based on a plain-English description of what the user wants to hear. It is designed as an applied AI portfolio project demonstrating retrieval-augmented generation, agentic orchestration, and reliability guardrails. It is suitable for personal use and classroom demonstration. It is not designed for production deployment, does not handle user authentication, and makes no guarantees about catalog completeness or content appropriateness.

---

## 3. How the System Works

When a user types "calm acoustic songs for late night studying," the system does the following:

**Retrieval (RAG):** The query is encoded into a 384-dimensional vector using `sentence-transformers/all-MiniLM-L6-v2`, a lightweight local model. That vector is compared against pre-computed embeddings for every song in the catalog using cosine similarity. The 100 most semantically similar songs become the candidate pool.

**Profile Extraction:** Gemini (`gemma-3-27b-it`) reads the query and outputs a structured JSON profile — preferred genre, mood, energy level, valence, danceability, acousticness, and tempo. Few-shot examples in the prompt calibrate Gemini's output to the music domain. This profile describes what the user wants numerically, which the scoring engine can act on.

**Scoring (original engine):** The same `score_song()` function from Module 1 scores each of the 100 candidates against the extracted profile using a weighted feature sum. Energy (25%) and valence (20%) carry the most weight, reflecting Spotify's published research on which audio features best predict listener satisfaction. Genre carries the least weight (5%) because the continuous features already encode genre implicitly.

**Output Guardrail:** Gemini reviews the top results and scores confidence from 0.0 to 1.0. If confidence is below 0.6, the agent appends the guardrail's feedback to the query and retries the full pipeline (max 2 attempts).

**Refinement:** The user may optionally describe a further adjustment ("more upbeat," "less acoustic"). Gemini updates the existing profile to reflect the refinement, and the scoring engine re-ranks the same 100 candidates. No new search is performed.

---

## 4. Data

**Catalog:** 81,343 tracks sourced from the [Kaggle Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). After deduplication by title and artist, the catalog spans 113 genres.

**Audio features used:** energy, valence, danceability, acousticness, tempo\_bpm, genre. All continuous features are provided by Spotify's audio analysis API and stored in the Kaggle dataset.

**Mood derivation:** The Kaggle dataset has no mood column. Mood is derived from energy and valence using threshold rules:

| Rule | Mood |
|---|---|
| energy > 0.7 and valence > 0.7 | happy |
| energy > 0.7 and valence < 0.4 | intense |
| energy < 0.4 and valence > 0.6 | chill |
| energy < 0.4 and valence < 0.4 | melancholic |
| otherwise | neutral |

53% of tracks fall into "neutral" because most songs have mid-range energy and valence. This is a known limitation.

**Embeddings:** Song text is formatted as `"{title} by {artist}. Genre: {genre}. Mood: {mood}. Energy: {energy}, Valence: {valence}."` and encoded with `all-MiniLM-L6-v2` (384 dimensions, ~175MB for the full catalog).

---

## 5. AI Components

| Component | Model | Purpose | Mock available |
|---|---|---|---|
| Input guardrail | `gemma-3-27b-it` | Reject non-music queries | Yes |
| Profile extraction | `gemma-3-27b-it` | NL query → JSON UserProfile | Yes |
| Profile refinement | `gemma-3-27b-it` | Update profile from follow-up | Yes |
| Output guardrail | `gemma-3-27b-it` | Score result quality 0–1 | Yes |
| Semantic search | `all-MiniLM-L6-v2` | Encode query and catalog | No (local) |

All Gemini calls accept `mock=True` for zero-cost testing. The scoring engine (`score_song()`) is deterministic Python — no model involved.

---

## 6. Strengths

- **Transparent scoring.** Every recommendation includes a plain-language explanation of why it was chosen ("energy is close to your preference," "matches your preferred genre"). No black box.
- **Natural language input.** Users describe what they want in plain English instead of specifying numeric audio features. Gemini handles the translation.
- **Agentic reliability.** The output guardrail catches bad result sets and triggers a retry rather than silently returning poor matches. The confidence score is surfaced to the user.
- **Two-stage refinement.** Users can narrow results with a follow-up prompt without re-running the expensive semantic search step.
- **Fully testable without API calls.** All Gemini interactions are mockable. The 31-test suite runs in seconds with zero API cost.
- **Large, real catalog.** 81k Spotify tracks with verified audio features provides genuine genre and mood diversity.

---

## 7. Limitations and Bias

- **Mood labels are crude.** Five derived mood categories cannot capture real emotional nuance. A blues song and a slow ambient track both land in "melancholic" despite feeling completely different.
- **Genre label inconsistency.** The Kaggle dataset uses 113 genre strings, but Gemini may extract genre names that don't exactly match catalog strings (e.g., "rap" vs. "hip-hop"). The genre pre-filter fails silently when strings don't match.
- **No popularity or quality signal.** A well-known classic and an obscure track with identical audio features score identically. The system cannot distinguish cultural significance.
- **No lyrics, language, or cultural awareness.** A user asking for Spanish-language music or explicitly lyrical hip-hop gets no special handling.
- **Static catalog.** The catalog is a snapshot from one Kaggle download. New releases are not included.
- **No user history persistence.** Liked and skipped signals are not saved between sessions. Every conversation starts from scratch.
- **Gemini profile hallucination risk.** If Gemini produces extreme values (energy = 2.0) or omits fields, defaults are applied silently. The user is not told their query was partially misunderstood.

---

## 8. Evaluation

**Unit tests:** 31 tests covering scoring math, pipeline integration, guardrail pass/fail, retry behavior, and error handling. All pass with zero API calls in mock mode.

**Evaluation harness:** 10 predefined query fixtures run in mock mode. 10/10 passed (≥5 results, confidence ≥0.6). Average confidence: 0.85.

**Live testing observations:**

- Before the genre pre-filter was added, "gangsta rap for a late night drive" returned metal and rock songs. The output guardrail scored 0.20 confidence — correctly identifying the mismatch — and the retry loop used the flag message to refine the query. The guardrail worked as designed.
- After the genre pre-filter was added, genre-specific queries consistently return genre-appropriate results.
- Refinement prompts ("more upbeat," "less acoustic," "faster tempo") produce measurably different profiles and result sets, confirming the refinement pipeline works end-to-end.

---

## 9. Future Work

- **Integrate Spotify API.** Replace the static Kaggle snapshot with live Spotify data. This enables real-time catalog updates, popularity signals, and user history via OAuth.
- **Persist user history.** Save liked/skipped signals to a local file between sessions so the profile improves with use.
- **Improve mood labels.** Replace threshold-derived mood with a small classifier trained on user annotations, or use Spotify's own mood/valence clusters.
- **Handle genre vocabulary mismatches.** Build a mapping between common genre synonyms (rap → hip-hop, r&b → soul, etc.) so the genre filter is robust to Gemini's vocabulary choices.
- **Add popularity weighting.** Weight `score_song()` output by a logarithmic popularity factor to surface well-regarded tracks over obscure ones when scores are close.

---

## 10. Reflection

The most instructive moment in building this system was discovering that the output guardrail correctly identified bad recommendations before I had added the genre pre-filter. The system returned metal songs for a hip-hop query, scored them at 0.20 confidence, and flagged the genre mismatch — all automatically. That was the guardrail working exactly as intended: not preventing bad results from being generated, but catching them before they reach the user and giving the agent enough information to try again.

This revealed something important about how to think about AI reliability. The goal is not to make the AI infallible on the first try — that's not achievable for a system that reasons over natural language. The goal is to make failures visible and recoverable. The confidence score makes failure explicit. The retry loop makes it recoverable. The explanation string makes it debuggable. Building those three things together is what distinguishes a reliable AI system from one that just happens to work most of the time.
