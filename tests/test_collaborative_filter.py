import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from recommender import load_songs
from collaborative_filter import CollaborativeFilter, UserKNNCollaborativeFilter, _rate_song, _ARCHETYPES

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


@pytest.fixture(scope="module")
def songs():
    return load_songs(_CSV)


@pytest.fixture(scope="module")
def fitted_cf(songs):
    return CollaborativeFilter(songs).fit()


# ── _rate_song ────────────────────────────────────────────────────────────────

def test_rate_song_perfect_match(songs):
    lofi_user = ("lofi", "chill", 0.42, True)
    midnight = next(s for s in songs if s["title"] == "Midnight Coding")
    score = _rate_song(lofi_user, midnight)
    assert score > 4.0, "near-perfect archetype match should score above 4"


def test_rate_song_mismatch(songs):
    rock_user = ("rock", "intense", 0.91, False)
    ambient = next(s for s in songs if s["title"] == "Spacewalk Thoughts")
    score = _rate_song(rock_user, ambient)
    assert score < 2.0, "opposite archetype should score below 2"


def test_rate_song_bounds(songs):
    for archetype in _ARCHETYPES:
        for song in songs:
            s = _rate_song(archetype, song)
            assert 0.0 <= s <= 5.0, f"rating out of [0,5]: {s}"


# ── fit ───────────────────────────────────────────────────────────────────────

def test_fit_returns_self(songs):
    cf = CollaborativeFilter(songs)
    result = cf.fit()
    assert result is cf


def test_fit_sets_components(fitted_cf):
    assert fitted_cf._H is not None
    assert fitted_cf._model is not None


def test_fit_h_shape(fitted_cf, songs):
    n_comp, n_songs = fitted_cf._H.shape
    assert n_songs == len(songs)
    assert 1 <= n_comp <= 4


# ── recommend ─────────────────────────────────────────────────────────────────

def test_recommend_returns_k_results(fitted_cf):
    recs = fitted_cf.recommend({"genre": "lofi", "mood": "chill", "energy": 0.4}, k=5)
    assert len(recs) == 5


def test_recommend_sorted_descending(fitted_cf):
    recs = fitted_cf.recommend({"genre": "pop", "mood": "happy", "energy": 0.8}, k=5)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_recommend_returns_song_dicts(fitted_cf):
    recs = fitted_cf.recommend({"genre": "rock", "mood": "intense", "energy": 0.9}, k=3)
    for song, score in recs:
        assert "title" in song
        assert "artist" in song
        assert isinstance(score, float)


def test_recommend_lofi_profile_prefers_lofi(fitted_cf, songs):
    recs = fitted_cf.recommend({"genre": "lofi", "mood": "chill", "energy": 0.38}, k=3)
    top_genres = [song["genre"] for song, _ in recs]
    assert "lofi" in top_genres, "lofi profile should surface at least one lofi song in top 3"


def test_recommend_unfitted_raises(songs):
    cf = CollaborativeFilter(songs)
    with pytest.raises(RuntimeError, match="fit"):
        cf.recommend({"genre": "pop", "mood": "happy", "energy": 0.8})


# ── UserKNNCollaborativeFilter ───────────────────────────────────────────────

def test_userknn_fit_returns_self(songs):
    userknn = UserKNNCollaborativeFilter(songs)
    result = userknn.fit()
    assert result is userknn


def test_userknn_recommend_returns_k_results(songs):
    userknn = UserKNNCollaborativeFilter(songs).fit()
    recs = userknn.recommend({"genre": "lofi", "mood": "chill", "energy": 0.4}, k=5)
    assert len(recs) == 5


def test_userknn_recommend_sorted_descending(songs):
    userknn = UserKNNCollaborativeFilter(songs).fit()
    recs = userknn.recommend({"genre": "pop", "mood": "happy", "energy": 0.8}, k=5)
    scores = [score for _, score in recs]
    assert scores == sorted(scores, reverse=True)


def test_userknn_recommend_returns_song_dicts(songs):
    userknn = UserKNNCollaborativeFilter(songs).fit()
    recs = userknn.recommend({"genre": "rock", "mood": "intense", "energy": 0.9}, k=3)
    for song, score in recs:
        assert "title" in song
        assert "artist" in song
        assert isinstance(score, float)


def test_userknn_lofi_profile_prefers_lofi(songs):
    userknn = UserKNNCollaborativeFilter(songs).fit()
    recs = userknn.recommend({"genre": "lofi", "mood": "chill", "energy": 0.38}, k=3)
    top_genres = [song["genre"] for song, _ in recs]
    assert "lofi" in top_genres, "lofi profile should surface at least one lofi song in top 3"


def test_userknn_unfitted_raises(songs):
    userknn = UserKNNCollaborativeFilter(songs)
    with pytest.raises(RuntimeError, match="fit"):
        userknn.recommend({"genre": "pop", "mood": "happy", "energy": 0.8})

# ── find_similar_songs ────────────────────────────────────────────────────────

def test_similar_excludes_query_song(fitted_cf, songs):
    song_id = songs[0]["id"]
    similar = fitted_cf.find_similar_songs(song_id, k=3)
    ids = [s["id"] for s, _ in similar]
    assert song_id not in ids


def test_similar_returns_k(fitted_cf, songs):
    similar = fitted_cf.find_similar_songs(songs[0]["id"], k=3)
    assert len(similar) == 3


def test_similar_cosine_in_range(fitted_cf, songs):
    # NMF factors are non-negative, so cosine similarity must be in [0, 1].
    similar = fitted_cf.find_similar_songs(songs[0]["id"], k=5)
    for _, sim in similar:
        assert 0.0 <= sim <= 1.0


def test_similar_sorted_descending(fitted_cf, songs):
    similar = fitted_cf.find_similar_songs(songs[1]["id"], k=5)
    sims = [sim for _, sim in similar]
    assert sims == sorted(sims, reverse=True)


def test_similar_unknown_id_raises(fitted_cf):
    with pytest.raises(ValueError, match="Unknown song id"):
        fitted_cf.find_similar_songs(song_id=9999)


def test_similar_unfitted_raises(songs):
    cf = CollaborativeFilter(songs)
    with pytest.raises(RuntimeError, match="fit"):
        cf.find_similar_songs(songs[0]["id"])
