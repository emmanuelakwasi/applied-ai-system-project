"""
VibeMatch Music Recommender — main entry point.

Three modes:
  default   — rule-based scoring (Step 1) + one-shot RAG synthesis (Step 2)
  --classic — rule-based scores only, no AI
  --reason  — multi-step reasoning agent: Claude calls tools and each
              intermediate step is printed as it happens

Usage (from project root):
    python src/main.py                                  # default profile, RAG synthesis
    python src/main.py "chill music for studying"       # natural language query
    python src/main.py --profile "Chill Lofi"           # preset profile
    python src/main.py --classic                        # scores only, no AI
    python src/main.py "late-night studying" --reason   # multi-step reasoning agent
"""

import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Allow direct imports of sibling modules (recommender, rag_pipeline)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Preset profiles and query → profile heuristic
# ---------------------------------------------------------------------------

SAMPLE_USER_PROFILES = {
    "High-Energy Pop":   {"genre": "pop",      "mood": "happy",   "energy": 0.8},
    "Chill Lofi":        {"genre": "lofi",     "mood": "chill",   "energy": 0.4},
    "Deep Intense Rock": {"genre": "rock",     "mood": "intense", "energy": 0.9},
}

# Keyword → nearest UserProfile for natural-language queries.
# The rule-based scorer needs a discrete profile; these heuristics bridge
# free-text queries to the closest matching profile so scores are meaningful.
_KEYWORD_PROFILES = [
    (["workout", "gym", "pump", "run", "sprint", "intense", "power"],
     {"genre": "rock",     "mood": "intense", "energy": 0.9}),
    (["chill", "relax", "study", "focus", "concentrate", "lofi", "quiet", "background"],
     {"genre": "lofi",     "mood": "chill",   "energy": 0.4}),
    (["happy", "pop", "upbeat", "morning", "fun", "cheerful", "dance"],
     {"genre": "pop",      "mood": "happy",   "energy": 0.8}),
    (["ambient", "sleep", "meditat", "dreamy", "space", "calm"],
     {"genre": "ambient",  "mood": "chill",   "energy": 0.28}),
    (["jazz", "coffee", "evening", "dinner", "relaxed", "laid-back"],
     {"genre": "jazz",     "mood": "relaxed", "energy": 0.37}),
    (["synthwave", "night", "drive", "moody", "dark", "late"],
     {"genre": "synthwave","mood": "moody",   "energy": 0.75}),
    (["indie", "rooftop", "summer", "breezy"],
     {"genre": "indie pop","mood": "happy",   "energy": 0.76}),
]


def _infer_profile(query: str) -> dict:
    """Map a natural language query to the nearest preset user profile."""
    q = query.lower()
    for keywords, profile in _KEYWORD_PROFILES:
        if any(kw in q for kw in keywords):
            return profile
    return SAMPLE_USER_PROFILES["High-Energy Pop"]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_DIV = "-" * 58


def _print_scores(label: str, recommendations: list) -> None:
    print(f"\n{_DIV}")
    print(f"  Step 1 | Rule-Based Scores  [{label}]")
    print(_DIV)
    for i, (song, score, explanation) in enumerate(recommendations, 1):
        print(f"  {i}. {song['title']} by {song['artist']}  (score {score:.2f})")
        print(f"     {explanation}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VibeMatch — rule-based scoring + AI synthesis"
    )
    parser.add_argument(
        "query", nargs="?", default="",
        help="Natural language music request (e.g. 'chill music for late-night studying')"
    )
    parser.add_argument(
        "--profile", choices=list(SAMPLE_USER_PROFILES),
        help="Use a preset profile instead of a natural language query"
    )
    parser.add_argument(
        "--classic", action="store_true",
        help="Show rule-based scores only — skip AI synthesis"
    )
    parser.add_argument(
        "--reason", action="store_true",
        help="Use multi-step reasoning agent — Claude calls tools and each step is printed"
    )
    args = parser.parse_args()

    csv_path = os.path.join(_ROOT_DIR, "data", "songs.csv")
    docs_paths = [
        os.path.join(_ROOT_DIR, "data", "genre_profiles.json"),
        os.path.join(_ROOT_DIR, "data", "activity_contexts.json"),
        os.path.join(_ROOT_DIR, "data", "artist_notes.json"),
    ]
    songs = load_songs(csv_path)

    # --- Resolve query and user profile ---
    if args.query:
        query = args.query
        user_prefs = _infer_profile(query)
        label = f'"{query}"'
    elif args.profile:
        query = f"music matching {args.profile} preferences"
        user_prefs = SAMPLE_USER_PROFILES[args.profile]
        label = args.profile
    else:
        query = "upbeat energetic pop music that's happy and danceable"
        user_prefs = SAMPLE_USER_PROFILES["High-Energy Pop"]
        label = "High-Energy Pop (default)"

    recommendations = recommend_songs(user_prefs, songs, k=5)

    # --reason mode: skip step 1 display; the reasoning agent calls score_songs itself
    if args.reason:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("  (Set ANTHROPIC_API_KEY to enable the reasoning agent)")
            return
        try:
            from reasoning import ReasoningAssistant
        except ImportError as e:
            print(f"  Reasoning dependencies not installed: {e}")
            return
        agent = ReasoningAssistant(
            csv_path=csv_path, docs_paths=docs_paths, api_key=api_key
        )
        agent.recommend(query)
        return

    # --- Step 1: rule-based scoring ---
    _print_scores(label, recommendations)

    if args.classic:
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  (Set ANTHROPIC_API_KEY to enable AI-powered synthesis)")
        return

    # --- Step 2: RAG synthesis ---
    try:
        from rag_pipeline import RAGAssistant
    except ImportError as e:
        print(f"  RAG dependencies not installed: {e}")
        return

    assistant = RAGAssistant(csv_path=csv_path, docs_paths=docs_paths, api_key=api_key)

    print(f"{_DIV}")
    print("  Step 2 | AI Synthesis  [Claude reasoning over retrieved evidence]")
    print(f"{_DIV}\n")

    assistant.recommend_with_scores(query, recommendations, stream=True)


if __name__ == "__main__":
    main()
