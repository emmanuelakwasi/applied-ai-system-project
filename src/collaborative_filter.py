"""
Collaborative filtering via NMF (Non-negative Matrix Factorization).

Workflow:
  1. A set of synthetic listener archetypes generates an implicit rating matrix.
  2. NMF factorises the matrix into latent user factors (W) and item factors (H).
  3. A new user profile is projected into latent space (cold-start via transform),
     and predicted ratings W_new @ H select the top recommendations.
  4. Item-item similarity is computed directly from the item latent factors (H).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity

# ── Synthetic listener archetypes ────────────────────────────────────────────
# (genre_pref, mood_pref, energy_pref, likes_acoustic)
# Broad coverage ensures NMF latent factors span the main taste dimensions.
_ARCHETYPES: List[Tuple[str, str, float, bool]] = [
    # Lo-fi / study
    ("lofi",      "chill",   0.40, True),
    ("lofi",      "chill",   0.35, True),
    ("lofi",      "focused", 0.38, True),
    ("ambient",   "chill",   0.28, True),
    ("ambient",   "focused", 0.30, True),
    # Workout / intense
    ("rock",      "intense", 0.91, False),
    ("rock",      "intense", 0.88, False),
    ("pop",       "intense", 0.93, False),
    # Upbeat / pop
    ("pop",       "happy",   0.82, False),
    ("pop",       "happy",   0.78, False),
    ("indie pop", "happy",   0.76, True),
    # Synthwave / night
    ("synthwave", "moody",   0.75, False),
    ("synthwave", "moody",   0.80, False),
    # Jazz / cafe
    ("jazz",      "relaxed", 0.37, True),
    ("jazz",      "relaxed", 0.40, True),
    # Mixed listeners
    ("indie pop", "relaxed", 0.55, True),
    ("lofi",      "happy",   0.55, True),
    ("pop",       "chill",   0.60, False),
]

_BASE_MAX = 7.0       # genre(3) + mood(2) + energy(2)
_ACOUSTIC_BONUS = 1.5


def _rate_song(archetype: Tuple[str, str, float, bool], song: Dict) -> float:
    """Compute a plausible implicit rating in [0, 5] for one archetype-song pair."""
    genre_pref, mood_pref, energy_pref, likes_acoustic = archetype
    raw = 0.0

    if song["genre"] == genre_pref:
        raw += 3.0
    if song["mood"] == mood_pref:
        raw += 2.0

    energy_gap = abs(energy_pref - song["energy"])
    raw += max(0.0, 1.0 - energy_gap) * 2.0

    if likes_acoustic:
        raw += song["acousticness"] * _ACOUSTIC_BONUS

    # Normalise against this archetype's actual maximum so every user can reach 5.0.
    # Using a fixed 8.5 constant would deflate non-acoustic users' ratings by ~18%.
    max_raw = _BASE_MAX + (_ACOUSTIC_BONUS if likes_acoustic else 0.0)
    return float(np.clip(raw * (5.0 / max_raw), 0.0, 5.0))


class CollaborativeFilter:
    """
    Collaborative filtering using NMF matrix factorization.

    ``recommend()`` uses model-based CF: a synthetic rating matrix is factorised
    into latent user (W) and item (H) factors; new users are cold-started by
    projecting their rating proxy into latent space via NMF.transform().

    ``find_similar_songs()`` is item-item CF: songs are compared by cosine
    similarity in the latent item space (columns of H).

    Usage::

        cf = CollaborativeFilter(songs).fit()
        recs    = cf.recommend({"genre": "lofi", "mood": "chill", "energy": 0.4})
        similar = cf.find_similar_songs(song_id=2, k=3)
    """

    def __init__(self, songs: List[Dict], n_components: int = 4) -> None:
        self.songs = songs
        self.n_components = n_components
        self._song_index: Dict[int, int] = {s["id"]: i for i, s in enumerate(songs)}
        self._model: NMF | None = None
        self._H: np.ndarray | None = None  # item latent factors (n_comp × n_songs)

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self) -> "CollaborativeFilter":
        """Build the synthetic rating matrix and fit NMF."""
        n_comp = min(self.n_components, len(_ARCHETYPES) - 1, len(self.songs) - 1)

        R = np.array(
            [[_rate_song(u, s) for s in self.songs] for u in _ARCHETYPES]
        )  # shape: (n_users, n_songs)

        self._model = NMF(n_components=n_comp, random_state=0, max_iter=500)
        self._model.fit_transform(R)
        self._H = self._model.components_  # shape: (n_comp, n_songs)
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def recommend(self, user_prefs: Dict, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Return the top-k songs predicted for a new user profile.

        Cold-start: builds a rating proxy vector from the profile, projects it
        into latent space with NMF.transform(), then scores all songs via W @ H.
        """
        if self._model is None or self._H is None:
            raise RuntimeError("Call fit() before recommend()")

        genre  = user_prefs.get("genre", "pop")
        mood   = user_prefs.get("mood", "happy")
        energy = float(user_prefs.get("energy", 0.7))
        likes_acoustic = genre in {"lofi", "ambient", "jazz", "indie pop"}

        proxy = (genre, mood, energy, likes_acoustic)
        r_new = np.array([[_rate_song(proxy, s) for s in self.songs]])  # (1, n_songs)

        w_new = self._model.transform(r_new)   # (1, n_comp)
        r_pred = (w_new @ self._H).flatten()   # (n_songs,)

        ranked = sorted(
            zip(self.songs, r_pred.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    def find_similar_songs(self, song_id: int, k: int = 3) -> List[Tuple[Dict, float]]:
        """Return the k most similar songs by cosine similarity in latent item space."""
        if self._H is None:
            raise RuntimeError("Call fit() before find_similar_songs()")

        idx = self._song_index.get(song_id)
        if idx is None:
            raise ValueError(f"Unknown song id: {song_id}")

        item_vecs = self._H.T  # (n_songs, n_comp)
        sims = cosine_similarity(item_vecs[idx : idx + 1], item_vecs).flatten()

        ranked = sorted(
            [(self.songs[i], float(sims[i])) for i in range(len(self.songs)) if i != idx],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]
