from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv

DATA_FLOW_MAP = """
Data flow:
Input -> User preferences such as favorite genre, favorite mood, and target energy.
Process -> Loop through every song loaded from the CSV, score each one using genre match, mood match, and energy similarity.
Output -> Sort songs by score and return the top k recommendations with short explanations.
""".strip()

MERMAID_FLOWCHART = """
flowchart TD
    A[User Preferences] --> B[Load songs.csv]
    B --> C[Select one song from CSV]
    C --> D[Compare song to user preferences]
    D --> E[Score song<br/>Genre + Mood + Energy Similarity]
    E --> F[Store song with score and explanation]
    F --> G{More songs left?}
    G -->|Yes| C
    G -->|No| H[Sort all scored songs by score]
    H --> I[Return Top K recommendations]
""".strip()

LOAD_SONGS_PROMPT = """
Implement the load_songs function in Python.
Use Python's csv module to read #file:data/songs.csv and return a list of dictionaries.
Each dictionary should include the song fields from the CSV, with numeric values converted to int or float where appropriate.
""".strip()

SCORE_SONG_PROMPT = """
Implement score_song(user_prefs, song) using this algorithm recipe:
- Award +1.0 point for a genre match.
- Award +1.0 point for a mood match.
- Award energy similarity points based on how close the song's energy is to the user's target energy.
- Calculate energy similarity as max(0.0, 1.0 - abs(user_prefs["energy"] - song["energy"])) * 4.0.
- Return the final numeric score.
""".strip()

RECOMMEND_SONGS_PROMPT = """
With #file:src/recommender.py as context, implement the most Pythonic way to loop through all songs,
calculate each song's score using score_song(user_prefs, song), and return the top k results sorted
from highest to lowest score.

Use a clean approach such as building scored tuples, sorting by score in descending order, and slicing
to the top k results.
""".strip()

POINT_WEIGHTING_PROMPT = """
Use the data in #file:data/songs.csv as context for designing a transparent song recommendation scoring system.

I am building a simple classroom music recommender. Please suggest a balanced point-weighting strategy for these song features:
- genre
- mood
- energy
- acousticness
- tempo_bpm
- valence
- danceability

The goal is to create a score that feels intuitive, not extreme. I want clear tradeoffs between exact category matches and close numeric matches.

Please answer these questions:
1. How many points should a Genre match be worth?
2. How many points should a Mood match be worth?
3. Should Genre count more than Mood, or should they be equal? Explain why.
4. How should near-matches in Energy be scored compared to exact matches in Genre or Mood?
5. Should Acousticness, Tempo, Valence, and Danceability have smaller bonus weights? If so, suggest reasonable point values.
6. Propose one complete sample formula with point values and a short explanation of why it is balanced for this dataset.

Please keep the recommendation system simple and interpretable, and base your suggestions on the kinds of songs and attributes that appear in the CSV.
""".strip()

GENRE_MATCH_POINTS = 1.0
MOOD_MATCH_POINTS = 1.0
MAX_ENERGY_SIMILARITY_POINTS = 4.0

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        ranked_songs = sorted(
            self.songs,
            key=lambda song: self._score_song(user, song),
            reverse=True,
        )
        return ranked_songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        parts = []
        if song.genre == user.favorite_genre:
            parts.append(
                f"genre match (+{GENRE_MATCH_POINTS:.1f})"
            )
        if song.mood == user.favorite_mood:
            parts.append(
                f"mood match (+{MOOD_MATCH_POINTS:.1f})"
            )

        energy_points = _energy_similarity_points(user.target_energy, song.energy)
        parts.append(
            f"energy closeness (+{energy_points:.2f})"
        )

        if not parts:
            return "This song was included as a lower-confidence match."
        return "Strong fit because of " + ", ".join(parts) + "."

    def _score_song(self, user: UserProfile, song: Song) -> float:
        score = 0.0

        if song.genre == user.favorite_genre:
            score += GENRE_MATCH_POINTS
        if song.mood == user.favorite_mood:
            score += MOOD_MATCH_POINTS

        score += _energy_similarity_points(user.target_energy, song.energy)
        return score

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file into a list of dictionaries."""
    songs: List[Dict] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            songs.append(
                {
                    "id": int(row["id"]),
                    "title": row["title"],
                    "artist": row["artist"],
                    "genre": row["genre"],
                    "mood": row["mood"],
                    "energy": float(row["energy"]),
                    "tempo_bpm": float(row["tempo_bpm"]),
                    "valence": float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                }
            )
    return songs


def score_song(user_prefs: Dict, song: Dict) -> float:
    """Compute a song's recommendation score for a given user profile."""
    score = 0.0

    if song["genre"] == user_prefs["genre"]:
        score += GENRE_MATCH_POINTS
    if song["mood"] == user_prefs["mood"]:
        score += MOOD_MATCH_POINTS

    score += _energy_similarity_points(user_prefs["energy"], song["energy"])
    return score

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Rank songs by score and return the top k with explanations."""
    scored_songs: List[Tuple[Dict, float, str]] = []
    for song in songs:
        score = score_song(user_prefs, song)
        explanation_parts = []

        if song["genre"] == user_prefs["genre"]:
            explanation_parts.append(f"genre match (+{GENRE_MATCH_POINTS:.1f})")
        if song["mood"] == user_prefs["mood"]:
            explanation_parts.append(f"mood match (+{MOOD_MATCH_POINTS:.1f})")

        energy_points = _energy_similarity_points(user_prefs["energy"], song["energy"])
        explanation_parts.append(f"energy closeness (+{energy_points:.2f})")

        explanation = "Strong fit because of " + ", ".join(explanation_parts) + "."
        scored_songs.append((song, score, explanation))

    scored_songs.sort(key=lambda item: item[1], reverse=True)
    return scored_songs[:k]


def _energy_similarity_points(target_energy: float, song_energy: float) -> float:
    """Convert energy closeness into a bounded similarity score."""
    difference = abs(target_energy - song_energy)
    closeness = max(0.0, 1.0 - difference)
    return closeness * MAX_ENERGY_SIMILARITY_POINTS
