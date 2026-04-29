from src.recommender import score_song, recommend_songs


def make_song(id=1, genre="pop", energy=0.5, valence=0.5,
              acousticness=0.5, danceability=0.5, tempo_bpm=110):
    return {
        "id": id, "genre": genre, "energy": energy,
        "valence": valence, "acousticness": acousticness,
        "danceability": danceability, "tempo_bpm": tempo_bpm,
    }


def make_prefs(genre="pop", energy=0.5, valence=0.5,
               acousticness=0.5, danceability=0.5, tempo_bpm=110,
               liked_ids=None, skipped_ids=None):
    return {
        "preferred_genre": genre,
        "preferred_energy": energy, "preferred_valence": valence,
        "preferred_acousticness": acousticness,
        "preferred_danceability": danceability,
        "preferred_tempo_bpm": tempo_bpm,
        "liked_ids": liked_ids or [],
        "skipped_ids": skipped_ids or [],
    }


# --- score_song ---

def test_genre_match_scores_higher_than_mismatch():
    prefs = make_prefs(genre="pop")
    match = make_song(genre="pop")
    miss = make_song(genre="lofi")
    score_match, _ = score_song(prefs, match)
    score_miss, _ = score_song(prefs, miss)
    assert score_match > score_miss


def test_liked_song_gets_score_boost():
    song = make_song(id=1, genre="lofi")
    prefs_no_like = make_prefs(genre="pop")
    prefs_liked = make_prefs(genre="pop", liked_ids=[1])

    score_base, _ = score_song(prefs_no_like, song)
    score_boosted, _ = score_song(prefs_liked, song)

    assert score_boosted > score_base


def test_liked_song_score_capped_at_one():
    prefs = make_prefs(liked_ids=[1])
    song = make_song(id=1, genre="pop", energy=0.5, valence=0.5,
                     acousticness=0.5, danceability=0.5, tempo_bpm=110)
    score, _ = score_song(prefs, song)
    assert score <= 1.0


def test_skipped_song_gets_score_penalty():
    prefs = make_prefs(skipped_ids=[1])
    song = make_song(id=1)
    score_penalized, _ = score_song(prefs, song)

    prefs_no_skip = make_prefs()
    score_base, _ = score_song(prefs_no_skip, song)

    assert score_penalized < score_base


def test_skipped_penalty_halves_score():
    prefs_skip = make_prefs(skipped_ids=[1])
    prefs_base = make_prefs()
    song = make_song(id=1)
    score_skip, _ = score_song(prefs_skip, song)
    score_base, _ = score_song(prefs_base, song)
    assert round(score_skip, 4) == round(score_base * 0.5, 4)


def test_score_song_always_returns_reasons():
    prefs = make_prefs(genre="classical")
    song = make_song(genre="pop")
    _, reasons = score_song(prefs, song)
    assert len(reasons) > 0


def test_perfect_feature_match_scores_near_one():
    prefs = make_prefs()
    song = make_song()
    score, _ = score_song(prefs, song)
    assert score >= 0.9


# --- recommend_songs ---

def test_recommend_songs_sorted_descending():
    prefs = make_prefs(genre="pop")
    songs = [
        make_song(id=1, genre="lofi"),
        make_song(id=2, genre="pop"),
        make_song(id=3, genre="pop", energy=0.3),
    ]
    results = recommend_songs(prefs, songs, k=3)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_songs_returns_k_results():
    prefs = make_prefs()
    songs = [make_song(id=i) for i in range(10)]
    results = recommend_songs(prefs, songs, k=4)
    assert len(results) == 4


def test_recommend_songs_k_larger_than_catalog():
    prefs = make_prefs()
    songs = [make_song(id=i) for i in range(3)]
    results = recommend_songs(prefs, songs, k=10)
    assert len(results) == 3


def test_recommend_songs_best_match_is_first():
    prefs = make_prefs(genre="pop")
    songs = [
        make_song(id=1, genre="lofi"),
        make_song(id=2, genre="pop"),
    ]
    results = recommend_songs(prefs, songs, k=2)
    assert results[0][0]["id"] == 2


# --- Adversarial / edge-case tests ---

def test_liked_and_skipped_conflict_surfaces_both_signals():
    song = make_song(id=1, genre="lofi")
    prefs = make_prefs(genre="pop", liked_ids=[1], skipped_ids=[1])
    _, reasons = score_song(prefs, song)
    reason_text = " ".join(reasons)
    assert "overrides prior like" in reason_text, "conflict between liked and skipped was not surfaced"