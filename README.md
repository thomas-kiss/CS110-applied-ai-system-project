# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

The recommender runs each song through a four-step scoring pipeline and returns
the top N matches ranked by relevance score.

---

### The Two Core Objects

**`Song`** — stores the raw features loaded from `data/songs.csv`

| Field | Type | Role |
|---|---|---|
| `id` | int | Unique identifier |
| `title` | string | Display only, not scored |
| `artist` | string | Display only, used for diversity in ranking |
| `genre` | string | Categorical — scored as match / no-match |
| `mood` | string | Categorical — scored as match / no-match |
| `energy` | float 0–1 | Continuous — scored by distance to preference |
| `valence` | float 0–1 | Continuous — scored by distance to preference |
| `danceability` | float 0–1 | Continuous — scored by distance to preference |
| `acousticness` | float 0–1 | Continuous — scored by distance to preference |
| `tempo_bpm` | float | Continuous — normalized before scoring |

**`UserProfile`** — mirrors the song's scoreable features and adds weights and
behavioral history

| Field | Type | Role |
|---|---|---|
| `name` | string | Display only |
| `preferred_genre` | string | Categorical target |
| `preferred_mood` | string | Categorical target |
| `preferred_energy` | float 0–1 | Numeric target |
| `preferred_valence` | float 0–1 | Numeric target |
| `preferred_danceability` | float 0–1 | Numeric target |
| `preferred_acousticness` | float 0–1 | Numeric target |
| `preferred_tempo_bpm` | float | Numeric target — normalized at score time |
| `liked_ids` | list[int] | Songs to boost after scoring |
| `skipped_ids` | list[int] | Songs to penalize after scoring |

---

### The Scoring Pipeline

Every song passes through four sequential steps to produce a final relevance
score between `0.0` and `1.0`.

#### Step 1 — Normalize Tempo

`tempo_bpm` is the only feature not already on a 0–1 scale.
Before scoring, both the song's tempo and the user's preferred tempo are
converted:
tempo_norm = (tempo_bpm - 60) / (160 - 60)

#### Step 2 — Per-Feature Similarity

For each **continuous** feature, similarity is measured by closeness to the
user's preference — not by raw value. A song at the exact preference scores
`1.0`; maximum distance scores `0.0`.
similarity = 1 - |song_value - user_preference|

For **categorical** features (genre, mood), the rule is binary:
score = 1.0  if song value matches user preference
score = 0.0  if it does not

#### Step 3 — Weighted Sum

Each feature similarity is multiplied by its weight and summed.
Weights reflect how strongly each dimension predicts listener satisfaction,
informed by how Spotify and SoundCloud weight their own audio features.
raw_score = (energy_sim       × 0.25)
+ (valence_sim      × 0.20)
+ (acousticness_sim × 0.20)
+ (danceability_sim × 0.15)
+ (tempo_sim        × 0.08)
+ (mood_score       × 0.07)
+ (genre_score      × 0.05)

> **Why these weights?**
> `energy` and `valence` are Spotify's two documented primary axes for audio
> analysis. `acousticness` strongly separates production styles that genre
> labels often miss. `genre` receives the lowest weight because the continuous
> features already implicitly encode it — giving it more weight would
> double-count the same signal.

#### Step 4 — Behavioral Adjustment

After the weighted sum, the user's listening history adjusts the score.
This runs last so behavioral signals modify relevance without corrupting the
feature similarity math.
if song_id in skipped_ids  →  score × 0.5
if song_id in liked_ids    →  score × 1.2  (capped at 1.0)

---

### Scoring vs. Ranking

The scoring rule evaluates **one song in isolation**. The ranking layer then sorts all scores and applies list-level rules — for example, no duplicate artists in the top 5 — before producing the final recommendation list. These are intentionally separate steps. A song can score highly but be displaced by a diversity rule, and the two reasons for exclusion stay independently debuggable.

**Per-song scoring:**

1. Compute per-feature similarity between song and user profile
2. Multiply each similarity by its weight and sum to a raw score
3. Apply behavioral adjustment (liked / skipped history)

**List-level ranking:**

4. Collect all scores and sort descending
5. Apply diversity filter (e.g. no duplicate artists in top 5)
6. Return top N recommendations

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

