# VibeMatch AI — Semantic Music Recommender

## Original Project
This project extends **VibeMatch 1.0**, a content-based music recommender built in Modules 1–3 of CodePath Ai110. The original system scored a hand-curated 20-track catalog against a hardcoded user profile using a six-feature weighted sum (energy, valence, acousticness, danceability, tempo, genre). It demonstrated the fundamentals of feature similarity scoring and behavioral signals (liked/skipped history), but required manually specifying numeric preferences and could only search 20 songs.

---

## What This Project Does
VibeMatch AI extends the original recommender into a full applied AI system. A user selects from a menu of preset listening profiles — or describes what they want in plain English — and the system returns tailored recommendations from an 81,000-track Spotify catalog. After seeing the initial results, the user can refine with a follow-up prompt ("more upbeat", "less acoustic") and the system re-ranks without re-searching.

The original `score_song()` and `recommend_songs()` functions are preserved as the reranking engine. All new functionality — preset profiles, semantic search, natural language understanding, guardrails, and agentic retry — wraps around them.

---

## System Architecture

![Architecture Diagram](assets/architecture.png)

### Component Flow

```mermaid
graph TD
    A[User Input: Preset or NL Description] --> B{Input Guardrail}
    B -- "Invalid (Non-Music)" --> C[Reject Query]
    B -- "Valid" --> D[RAG Retrieval: all-MiniLM-L6-v2]
    
    D --> E[100 Semantic Candidates]
    
    E --> F{Profile Extraction}
    F -- "Preset" --> G[Hardcoded Values]
    F -- "Custom" --> H[Gemini Few-Shot Extraction]
    
    G --> I[Scoring Engine: original score_song]
    H --> I
    
    I --> J{Output Guardrail}
    J -- "Confidence < 0.6" --> K[Agentic Retry: Refine & Search]
    K --> D
    J -- "Confidence >= 0.6" --> L[Top 10 Results]
    
    L --> M[User Refinement Prompt]
    M --> N[Profile Update: Gemini]
    N --> I