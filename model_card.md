# Model Card: Music Recommender Simulation

## 1. Model Name

**VibeMatch 1.0**

---

## 2. Intended Use

VibeMatch suggests up to five songs from a 20-track catalog based on a user's stated audio preferences and listening history. It is designed for classroom exploration of how content-based recommendation systems work, not for use with real listeners or a production catalog. It assumes the user can describe their taste numerically (e.g., preferred energy level, acousticness) and that their liked and skipped history is available.

---

## 3. How the Model Works

Imagine you told a friend: "I like upbeat, produced-sounding songs around 120 BPM — not too mellow, not too acoustic." A good friend would mentally scan their music library and hand you songs that fit that description. VibeMatch does the same thing, but with math.

For every song in the catalog, the system measures how close that song is to your stated preferences across seven dimensions: energy, emotional tone (valence), how acoustic or produced it sounds, how danceable it is, tempo, mood, and genre. Songs that are close to your preferences score near 1.0; songs that are far away score near 0.0.

The dimensions do not all count equally. Energy and emotional tone matter most because research from major streaming platforms shows these two features best predict whether a listener will enjoy a song. Genre matters least — partly because the other features already imply it, and partly because genre labels are often too broad to be useful on their own.

Finally, the system checks your history. If you previously liked a song, its score gets a small boost. If you skipped it, its score gets cut in half. If a song is in both your liked and skipped list, the skip wins — but the system now tells you that happened instead of silently ignoring the conflict.

---

## 4. Data

The catalog contains 20 songs stored in `data/songs.csv`. Each song was hand-authored with realistic audio feature values across 11 genres: pop, lo-fi, rock, ambient, jazz, synthwave, hip-hop, R&B, classical, country, metal, EDM, reggae, latin, folk, and blues. Moods covered include happy, chill, intense, focused, relaxed, moody, melancholic, romantic, peaceful, nostalgic, angry, energetic, uplifting, and sad.

The catalog was expanded from 10 to 20 songs during development to improve genre and mood coverage. However, it is still small and hand-curated, which means it reflects the taste assumptions of whoever built it. Genres like K-pop, Afrobeats, classical Indian, and most non-English-language music are absent entirely.

---

## 5. Strengths

- **Transparent scoring.** Every recommendation comes with a plain-language explanation of why it was chosen (e.g., "energy is close to your preference," "matches your preferred mood"). There is no black box.
- **Behavioral signals work correctly.** Liked songs get a meaningful boost; skipped songs are penalized. Conflicts between the two are now surfaced rather than silently resolved.
- **Feature weights are Spotify-informed.** Energy and valence are the two features most predictive of listener satisfaction according to Spotify's own published research. The weights reflect that.
- **Well-suited for users with clear audio preferences.** When a user's continuous feature preferences closely match a song's actual audio profile — as with the baseline Alex profile — the top results feel intuitively correct.

---

## 6. Limitations and Bias

- **Genre labels carry almost no weight.** Genre is only 5% of the score. A user who says they want pop but whose continuous preferences point toward folk will get folk recommendations with no explanation that their stated genre preference was overruled. This is intentional and Spotify-aligned, but can feel like the system ignored the user.
- **No collaborative filtering.** The system cannot learn from what similar users enjoy. It has no concept of "people who liked X also liked Y."
- **Static, recency-blind history.** A skip from six months ago counts exactly the same as a skip from yesterday. Real platforms weight recent signals more heavily.
- **Underrepresented genres and moods.** With 20 songs across 16+ genres, most genres have one or two representatives. Users whose taste lives outside pop, lo-fi, and rock will find poor matches.
- **No lyric or language awareness.** Two songs with identical audio features but different languages, themes, or cultural origins are treated as equivalent. A user who only listens to Spanish-language music gets no special handling.
- **Catalog bias.** The 20 songs were selected by the developers, which means the catalog implicitly reflects certain cultural and genre assumptions. Users outside those assumptions are underserved by design.

---

## 7. Evaluation

Three user profiles were tested and their outputs reviewed manually:

**Alex (baseline)** — A pop/happy listener with moderately high energy preferences and a history of liking two specific songs. Rooftop Lights and Fuego Lento both scored 1.00 after the liked-song boost, and the remaining three results were all energetic, produced-sounding tracks. Results matched expectations.

**Jordan (adversarial — genre vs. features)** — Declared pop/happy but set all continuous preferences in the opposite direction (quiet, acoustic, slow, sad). The top five results were all folk, classical, and ambient tracks — zero pop songs appeared. This confirmed the weight structure is working as intended and is consistent with how Spotify handles genre vs. audio similarity.

**Riley (adversarial — behavioral conflict)** — The same two song IDs appeared in both `liked_ids` and `skipped_ids`. Before the bug fix, the liked signal was silently discarded. After the fix, the conflict appears in the reason string. Both conflicted songs were excluded from the top five, which is the correct outcome.

A pytest suite of 13 unit tests covers liked/skipped penalties and boosts, score capping, sort order, result count, and the behavioral conflict case.

---

## 8. Future Work

- **Recency weighting for behavioral signals.** More recent skips and likes should count more than older ones. A simple decay function on the `liked_ids` and `skipped_ids` signals would significantly improve the model for returning users.
- **Collaborative filtering.** Adding a "users who liked this also liked" layer would let the system surface songs a user would not have described in their preferences but would genuinely enjoy.
- **Larger and more diverse catalog.** Twenty songs is a proof of concept. A real catalog would need thousands of tracks with better genre and cultural coverage.
- **Dynamic weight learning.** Instead of fixed weights, the system could adjust how much it trusts each feature based on a user's history — e.g., if a user consistently skips high-acousticness songs, that feature's weight could increase automatically.
- **Diversity enforcement.** The current ranking can return multiple songs by the same artist. A de-duplication pass on artist names in the top K would improve result variety.
- **Conflict resolution policy.** When a song appears in both `liked_ids` and `skipped_ids`, skip currently always wins. A better policy would use the timestamp of each signal so the most recent one wins.

---

## 9. Personal Reflection

The most surprising thing was how little the genre label mattered once continuous features were in play. A user who ticks "pop" on their profile can end up with folk recommendations and the system is technically correct — their audio preferences describe a folk listener. That gap between what a user says and what the model infers is not a bug, but it reveals something important: recommenders make assumptions on behalf of users, and those assumptions are invisible unless you specifically go looking for them. Building this system made me look at Spotify's "Discover Weekly" differently. The recommendations feel personal, but they are the output of weights someone else chose on features someone else defined — and the system will never tell you when your stated preference was quietly overruled.
