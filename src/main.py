"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    user_prefs = {
        "name": "Alex",
        "preferred_genre": "pop",
        "preferred_mood": "happy",
        "preferred_energy": 0.75,        # moderately high — prefers upbeat but not extreme
        "preferred_valence": 0.68,       # leans positive without requiring pure euphoria
        "preferred_danceability": 0.72,  # likes groovy/rhythmic tracks
        "preferred_acousticness": 0.28,  # favors produced sound; disfavors heavy acoustic
        "preferred_tempo_bpm": 118,      # comfortable mid-fast tempo (pop/indie range)
        "liked_ids": [10, 18],           # Rooftop Lights (indie pop), Fuego Lento (latin)
        "skipped_ids": [13, 19],         # Moonlit Sonata (classical), Pine Road (folk/sad)
    }

    # --- ADVERSARIAL PROFILE 1: "Genre-Locked, Feature-Opposite" ---
    # Probes: does a declared genre preference actually steer recommendations,
    # or do continuous features (80% of the score) drown it out?
    # This user claims to want pop/happy but every continuous preference points
    # at a slow, quiet, acoustic, melancholic folk song.  The genre/mood weights
    # are only 12% combined, so the dataset's actual pop songs (high energy,
    # high danceability, fast tempo) are expected to rank BELOW tracks like
    # Pine Road or Moonlit Sonata — the opposite of what the user said they want.
    # user_prefs1 = {
    #     "name": "Jordan (genre-locked, feature-opposite)",
    #     "preferred_genre": "pop",        # says they want pop …
    #     "preferred_mood": "happy",       # … and happy songs …
    #     "preferred_energy": 0.10,        # … but prefers near-silence
    #     "preferred_valence": 0.10,       # … and very sad/dark tone
    #     "preferred_danceability": 0.10,  # … and non-danceable
    #     "preferred_acousticness": 0.95,  # … and strongly acoustic
    #     "preferred_tempo_bpm": 65,       # … and very slow (near normalization floor)
    #     "liked_ids": [1],                # previously liked Sunrise City (pop/happy)
    #     "skipped_ids": [],
    # }

    # --- ADVERSARIAL PROFILE 2: "The Self-Contradiction" ---
    # Probes: what happens when the same song ID appears in both liked_ids AND
    # skipped_ids?  The scoring logic (recommender.py:119-124) checks skipped
    # first with an `elif`, so the skip penalty (* 0.5) always silently wins;
    # the liked boost (* 1.2) is never applied and no warning is surfaced.
    # Songs 1 and 5 (Sunrise City, Gym Hero) would naturally score well for a
    # pop/happy user, so the silent skip penalty meaningfully depresses them
    # without the user (or system) ever being told the conflict existed.
    # user_prefs2 = {
    #     "name": "Riley (conflicting behavioral signals)",
    #     "preferred_genre": "pop",
    #     "preferred_mood": "happy",
    #     "preferred_energy": 0.80,
    #     "preferred_valence": 0.80,
    #     "preferred_danceability": 0.80,
    #     "preferred_acousticness": 0.10,
    #     "preferred_tempo_bpm": 125,
    #     "liked_ids":   [1, 5],   # Sunrise City, Gym Hero — also in skipped_ids!
    #     "skipped_ids": [1, 5],   # same IDs: skip silently wins; liked boost never fires
    # }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
