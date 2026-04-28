"""
Unit tests for ReasoningAssistant tool implementations.

Each tool method is tested directly — no API calls are made.
The ReasoningAssistant is instantiated with a dummy API key; the client
object is created but never invoked.
"""

import re
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("sklearn", reason="scikit-learn required")

_DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH      = os.path.join(_DATA, "songs.csv")
GENRE_PATH    = os.path.join(_DATA, "genre_profiles.json")
ACTIVITY_PATH = os.path.join(_DATA, "activity_contexts.json")
ARTIST_PATH   = os.path.join(_DATA, "artist_notes.json")


@pytest.fixture(scope="module")
def agent():
    from reasoning import ReasoningAssistant
    return ReasoningAssistant(
        csv_path=CSV_PATH,
        docs_paths=[GENRE_PATH, ACTIVITY_PATH, ARTIST_PATH],
        api_key="test-key-no-calls-made",
    )


# ---------------------------------------------------------------------------
# score_songs tool
# ---------------------------------------------------------------------------

class TestToolScoreSongs:
    def test_returns_five_ranked_entries(self, agent):
        result = agent._tool_score_songs({"genre": "lofi", "mood": "chill", "energy": 0.4})
        numbered = [l for l in result.splitlines() if re.match(r"\s+[1-5]\.", l)]
        assert len(numbered) == 5

    def test_top_song_for_chill_profile_is_lofi(self, agent):
        result = agent._tool_score_songs({"genre": "lofi", "mood": "chill", "energy": 0.4})
        first_song_line = [l for l in result.splitlines() if re.match(r"\s+1\.", l)][0]
        assert "Midnight Coding" in first_song_line or "Library Rain" in first_song_line

    def test_top_song_for_rock_profile_is_storm_runner(self, agent):
        result = agent._tool_score_songs({"genre": "rock", "mood": "intense", "energy": 0.9})
        first_song_line = [l for l in result.splitlines() if re.match(r"\s+1\.", l)][0]
        assert "Storm Runner" in first_song_line

    def test_scores_are_decimal_numbers(self, agent):
        result = agent._tool_score_songs({"genre": "pop", "mood": "happy", "energy": 0.8})
        scores = re.findall(r"score \d+\.\d{2}", result)
        assert len(scores) == 5

    def test_each_entry_includes_valence_and_acousticness(self, agent):
        result = agent._tool_score_songs({"genre": "lofi", "mood": "chill", "energy": 0.4})
        assert "valence=" in result
        assert "acoustic=" in result


# ---------------------------------------------------------------------------
# retrieve_songs tool
# ---------------------------------------------------------------------------

class TestToolRetrieveSongs:
    def test_default_returns_three_results(self, agent):
        result = agent._tool_retrieve_songs({"query": "chill study music"})
        song_lines = [l for l in result.splitlines() if re.match(r"\s+\d+\.", l)]
        assert len(song_lines) == 3

    def test_k_parameter_respected(self, agent):
        result = agent._tool_retrieve_songs({"query": "workout gym energy", "k": 2})
        song_lines = [l for l in result.splitlines() if re.match(r"\s+\d+\.", l)]
        assert len(song_lines) == 2

    def test_chill_query_surfaces_lofi_genre(self, agent):
        result = agent._tool_retrieve_songs({"query": "late-night chill lo-fi studying"})
        assert "lofi" in result.lower()

    def test_intense_query_surfaces_rock_or_intense(self, agent):
        result = agent._tool_retrieve_songs({"query": "intense workout high energy pump"})
        assert "rock" in result.lower() or "intense" in result.lower()

    def test_result_includes_energy_and_acousticness(self, agent):
        result = agent._tool_retrieve_songs({"query": "ambient sleep"})
        assert "energy=" in result
        assert "acoustic=" in result


# ---------------------------------------------------------------------------
# get_genre_profile tool
# ---------------------------------------------------------------------------

class TestToolGetGenreProfile:
    def test_lofi_query_returns_lofi_content(self, agent):
        result = agent._tool_get_genre_profile({"genre": "lo-fi"})
        assert "lo-fi" in result.lower() or "lofi" in result.lower()
        assert len(result) > 200

    def test_ambient_query_returns_ambient_content(self, agent):
        result = agent._tool_get_genre_profile({"genre": "ambient"})
        assert "ambient" in result.lower()

    def test_rock_query_returns_rock_content(self, agent):
        result = agent._tool_get_genre_profile({"genre": "rock"})
        assert "rock" in result.lower()

    def test_unknown_genre_returns_string_not_exception(self, agent):
        result = agent._tool_get_genre_profile({"genre": "made-up-genre-zzzz"})
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# get_activity_context tool
# ---------------------------------------------------------------------------

class TestToolGetActivityContext:
    def test_studying_returns_focus_content(self, agent):
        result = agent._tool_get_activity_context({"activity": "studying"})
        assert "study" in result.lower() or "focus" in result.lower() or "cognitive" in result.lower()
        assert len(result) > 200

    def test_hiit_returns_workout_content(self, agent):
        result = agent._tool_get_activity_context({"activity": "HIIT workout"})
        assert "hiit" in result.lower() or "interval" in result.lower() or "workout" in result.lower()

    def test_sleep_returns_sleep_content(self, agent):
        result = agent._tool_get_activity_context({"activity": "sleep"})
        assert "sleep" in result.lower() or "wind-down" in result.lower() or "parasympathetic" in result.lower()

    def test_unknown_activity_returns_string_not_exception(self, agent):
        result = agent._tool_get_activity_context({"activity": "totally-unknown-activity-xyz"})
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# get_artist_notes tool
# ---------------------------------------------------------------------------

class TestToolGetArtistNotes:
    def test_loroom_returns_loroom_content(self, agent):
        result = agent._tool_get_artist_notes({"artist": "LoRoom"})
        assert "loroom" in result.lower() or "LoRoom" in result
        assert len(result) > 200

    def test_voltline_returns_rock_content(self, agent):
        result = agent._tool_get_artist_notes({"artist": "Voltline"})
        assert "voltline" in result.lower() or "Storm Runner" in result

    def test_orbit_bloom_returns_ambient_content(self, agent):
        result = agent._tool_get_artist_notes({"artist": "Orbit Bloom"})
        assert "ambient" in result.lower() or "orbit" in result.lower()

    def test_unknown_artist_returns_string_not_exception(self, agent):
        result = agent._tool_get_artist_notes({"artist": "Totally Unknown Artist 9999"})
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# Tool dispatch routing
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_dispatch_score_songs_returns_nonempty(self, agent):
        result = agent._dispatch(
            "score_songs", {"genre": "lofi", "mood": "chill", "energy": 0.4}, step=1
        )
        assert len(result) > 50

    def test_dispatch_unknown_tool_returns_error_string(self, agent):
        result = agent._dispatch("nonexistent_tool", {}, step=1)
        assert "Unknown tool" in result

    def test_dispatch_retrieve_songs_returns_nonempty(self, agent):
        result = agent._dispatch(
            "retrieve_songs", {"query": "chill music", "k": 2}, step=1
        )
        assert len(result) > 50
