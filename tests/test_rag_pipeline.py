"""
Tests for the RAG pipeline (retrieval layer only — no live API calls).
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

pytest.importorskip("sklearn", reason="scikit-learn required for RAG tests")

import json
from rag_pipeline import (
    SongKnowledgeBase, DocumentKnowledgeBase,
    _song_to_search_text, _format_song_line,
)


@pytest.fixture(scope="module")
def kb():
    return SongKnowledgeBase(CSV_PATH)


class TestSongKnowledgeBase:
    def test_loads_all_songs(self, kb):
        assert len(kb.all_songs) == 10

    def test_retrieve_returns_k_results(self, kb):
        results = kb.retrieve("upbeat pop music", k=3)
        assert len(results) == 3

    def test_retrieve_default_k(self, kb):
        results = kb.retrieve("chill lofi study")
        assert len(results) == 3

    def test_retrieve_k_larger_than_catalog(self, kb):
        # Should return all songs without error
        results = kb.retrieve("music", k=20)
        assert len(results) == 10

    def test_chill_query_returns_lofi(self, kb):
        results = kb.retrieve("chill lofi studying focus", k=3)
        genres = [s["genre"] for s in results]
        assert "lofi" in genres, f"Expected lofi in top-3 for chill query, got {genres}"

    def test_intense_query_returns_rock_or_pop(self, kb):
        results = kb.retrieve("intense high energy workout gym", k=3)
        moods = [s["mood"] for s in results]
        assert "intense" in moods, f"Expected intense mood in top-3, got {moods}"

    def test_acoustic_query_returns_acoustic_songs(self, kb):
        results = kb.retrieve("acoustic relaxed jazz coffee", k=3)
        # At least one result should have high acousticness
        acousticness_values = [s["acousticness"] for s in results]
        assert any(v > 0.5 for v in acousticness_values), (
            f"Expected acoustic song in results, got acousticness={acousticness_values}"
        )

    def test_retrieve_returns_dicts_with_required_keys(self, kb):
        required = {"id", "title", "artist", "genre", "mood", "energy",
                    "tempo_bpm", "valence", "danceability", "acousticness"}
        for song in kb.retrieve("anything", k=1):
            assert required.issubset(song.keys())

    def test_songs_have_correct_types(self, kb):
        for s in kb.all_songs:
            assert isinstance(s["id"], int)
            assert isinstance(s["energy"], float)
            assert isinstance(s["title"], str)


class TestHelpers:
    def test_song_to_search_text_contains_genre_and_mood(self):
        song = {
            "title": "Test Song", "artist": "Artist", "genre": "rock",
            "mood": "intense", "energy": 0.9, "tempo_bpm": 150,
            "valence": 0.5, "danceability": 0.6, "acousticness": 0.1
        }
        text = _song_to_search_text(song)
        assert "rock" in text
        assert "intense" in text
        assert "high-energy" in text

    def test_format_song_line_includes_title_and_artist(self):
        song = {
            "title": "Sunrise City", "artist": "Neon Echo", "genre": "pop",
            "mood": "happy", "energy": 0.82, "tempo_bpm": 118,
            "valence": 0.84, "danceability": 0.79, "acousticness": 0.18
        }
        line = _format_song_line(song)
        assert "Sunrise City" in line
        assert "Neon Echo" in line
        assert "pop" in line


# ---------------------------------------------------------------------------
# DocumentKnowledgeBase tests
# ---------------------------------------------------------------------------

class TestDocumentKnowledgeBase:
    def test_empty_paths_returns_zero_count(self):
        kb = DocumentKnowledgeBase([])
        assert kb.count == 0

    def test_retrieve_returns_empty_list_when_no_docs(self):
        kb = DocumentKnowledgeBase([])
        assert kb.retrieve("any query") == []

    def test_missing_file_is_skipped_gracefully(self, tmp_path):
        nonexistent = str(tmp_path / "does_not_exist.json")
        kb = DocumentKnowledgeBase([nonexistent])
        assert kb.count == 0
        assert kb.retrieve("query") == []

    def test_loads_docs_from_valid_json(self, tmp_path):
        docs = [{"id": "d1", "title": "Lo-fi Profile", "tags": ["lofi"], "content": "lo-fi study chill hip hop focus"}]
        p = tmp_path / "docs.json"
        p.write_text(json.dumps(docs))
        kb = DocumentKnowledgeBase([str(p)])
        assert kb.count == 1

    def test_retrieve_returns_most_relevant_doc(self, tmp_path):
        docs = [
            {"id": "d1", "title": "Lo-fi Profile",  "tags": ["lofi"],  "content": "lo-fi hip hop study chill focus quiet"},
            {"id": "d2", "title": "Rock Profile",    "tags": ["rock"],  "content": "rock intense workout gym energy high"},
        ]
        p = tmp_path / "docs.json"
        p.write_text(json.dumps(docs))
        kb = DocumentKnowledgeBase([str(p)])
        result = kb.retrieve("chill study music", k=1)
        assert result[0]["id"] == "d1"

    def test_retrieve_respects_k(self, tmp_path):
        docs = [
            {"id": f"d{i}", "title": f"Doc {i}", "tags": [], "content": f"content {i} music"}
            for i in range(5)
        ]
        p = tmp_path / "docs.json"
        p.write_text(json.dumps(docs))
        kb = DocumentKnowledgeBase([str(p)])
        assert len(kb.retrieve("music", k=3)) == 3

    def test_loads_from_real_genre_profiles(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "genre_profiles.json")
        kb = DocumentKnowledgeBase([path])
        assert kb.count == 7

    def test_genre_query_returns_matching_profile(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "genre_profiles.json")
        kb = DocumentKnowledgeBase([path])
        results = kb.retrieve("late-night studying chill lo-fi", k=1)
        assert "lo-fi" in results[0]["title"].lower() or "lofi" in " ".join(results[0]["tags"])

    def test_activity_query_returns_relevant_context(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "activity_contexts.json")
        kb = DocumentKnowledgeBase([path])
        results = kb.retrieve("high-intensity interval training", k=1)
        assert "hiit" in results[0]["title"].lower() or "hiit" in " ".join(results[0]["tags"]).lower()
