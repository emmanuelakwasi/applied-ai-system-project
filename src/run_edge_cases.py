from src.recommender import load_songs, recommend_songs


EDGE_CASE_PROFILES = {
    "Pop + Sad + High Energy": {"genre": "pop", "mood": "sad", "energy": 0.9},
    "Classical + Chill": {"genre": "classical", "mood": "chill", "energy": 0.4},
    "Lofi + Intense + Medium Energy": {"genre": "lofi", "mood": "intense", "energy": 0.4},
    "Rock + Chill + High Energy": {"genre": "rock", "mood": "chill", "energy": 0.9},
    "Ambient + Intense + Low Energy": {"genre": "ambient", "mood": "intense", "energy": 0.28},
    "Synthwave + Happy + Mid-High Energy": {"genre": "synthwave", "mood": "happy", "energy": 0.75},
    "Jazz + Relaxed + Negative Energy": {"genre": "jazz", "mood": "relaxed", "energy": -0.5},
    "Pop + Happy + Energy 2.0": {"genre": "pop", "mood": "happy", "energy": 2.0},
    "Blank Genre/Mood + Energy 0.76": {"genre": "", "mood": "", "energy": 0.76},
    "Indie Pop + Happy + Energy 0.79": {"genre": "indie pop", "mood": "happy", "energy": 0.79},
}


def main() -> None:
    songs = load_songs("data/songs.csv")

    for profile_name, user_prefs in EDGE_CASE_PROFILES.items():
        print(f"\n=== {profile_name} ===")
        recommendations = recommend_songs(user_prefs, songs, k=5)

        for index, (song, score, explanation) in enumerate(recommendations, start=1):
            print(f"{index}. {song['title']} by {song['artist']}")
            print(f"   Score: {score:.2f}")
            print(f"   Reasons: {explanation}")


if __name__ == "__main__":
    main()
