"""
Interactive CLI for the VibeMatch RAG assistant.

Usage:
    python src/rag_cli.py

Requires:
    ANTHROPIC_API_KEY environment variable set
    pip install anthropic scikit-learn numpy
"""

import os
import sys

# Allow running from repo root or from src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from rag_pipeline import RAGAssistant

CSV_PATH = os.path.join(_ROOT_DIR, "data", "songs.csv")
DOCS_PATHS = [
    os.path.join(_ROOT_DIR, "data", "genre_profiles.json"),
    os.path.join(_ROOT_DIR, "data", "activity_contexts.json"),
    os.path.join(_ROOT_DIR, "data", "artist_notes.json"),
]

BANNER = """
===========================================
     VibeMatch RAG Music Assistant
  Powered by Claude + TF-IDF Retrieval
==========================================

Example queries:
  • recommend upbeat music for a morning workout
  • explain why "Midnight Coding" is good for studying
  • classify the mood of "Night Drive Loop"
  • debug my playlist: Sunrise City → Storm Runner → Library Rain

Type 'quit' or 'exit' to stop.
"""

DIVIDER = "-" * 50


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    print(BANNER)

    try:
        assistant = RAGAssistant(csv_path=CSV_PATH, docs_paths=DOCS_PATHS)
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    print("Assistant ready. Ask anything about the music catalog.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Show which songs were retrieved
        retrieved = assistant.kb.retrieve(query, k=3)
        print(f"\n{DIVIDER}")
        print("Retrieved songs (TF-IDF similarity):")
        for s in retrieved:
            print(f"  [{s['genre']}/{s['mood']}] \"{s['title']}\" by {s['artist']} — energy={s['energy']}")
        print(f"{DIVIDER}")
        print("VibeMatch AI: ", end="", flush=True)

        try:
            assistant.ask(query, k=3, stream=True)
        except Exception as e:
            print(f"\nError calling Claude API: {e}")

        print()


if __name__ == "__main__":
    main()
