"""
VibeMatch AI Evaluation Harness.

Default mode  — runs 4 test queries through the full multi-source pipeline and
measures three quality criteria per response:
  cites_top_song  — Claude mentions the expected #1 rule-based pick by name
  cites_score     — Response contains at least one decimal score citation
  confidence_ok   — Claude's CONFIDENCE line parses to a value >= 0.5

--compare mode — runs one query in both single-source (songs.csv only) and
multi-source (songs.csv + genre profiles + activity guides + artist notes) modes
and prints a side-by-side depth metric comparison to quantify improvement.

Usage:
    python src/eval.py              # 4-case quality eval (multi-source)
    python src/eval.py --compare    # single vs. multi-source comparison

Requires ANTHROPIC_API_KEY. Default mode: ~4 API calls. Compare mode: ~2 calls.
"""

import argparse
import os
import re
import sys

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

from recommender import load_songs, recommend_songs
from rag_pipeline import RAGAssistant, _parse_confidence

CSV_PATH = os.path.join(_ROOT_DIR, "data", "songs.csv")
DOCS_PATHS = [
    os.path.join(_ROOT_DIR, "data", "genre_profiles.json"),
    os.path.join(_ROOT_DIR, "data", "activity_contexts.json"),
    os.path.join(_ROOT_DIR, "data", "artist_notes.json"),
]

EVAL_CASES = [
    {
        "label":        "chill/lo-fi query",
        "query":        "chill music for late-night studying",
        "user_prefs":   {"genre": "lofi", "mood": "chill", "energy": 0.4},
        "expected_top": "Midnight Coding",
    },
    {
        "label":        "intense/rock query",
        "query":        "high-intensity interval training playlist",
        "user_prefs":   {"genre": "rock", "mood": "intense", "energy": 0.9},
        "expected_top": "Storm Runner",
    },
    {
        "label":        "ambient/sleep query",
        "query":        "ambient music for deep sleep",
        "user_prefs":   {"genre": "ambient", "mood": "chill", "energy": 0.28},
        "expected_top": "Spacewalk Thoughts",
    },
    {
        "label":        "pop/happy query",
        "query":        "happy upbeat pop for a morning run",
        "user_prefs":   {"genre": "pop", "mood": "happy", "energy": 0.8},
        "expected_top": "Sunrise City",
    },
]

# Query used for the single vs. multi-source comparison
COMPARE_CASE = EVAL_CASES[0]

_SCORE_RE = re.compile(r"\b\d\.\d{2}\b")
_DIV = "-" * 60
_HDR = "=" * 60

# Attributes mentioned in song CSV — a multi-source response should cite more of these
_SONG_ATTRS = ["genre", "mood", "energy", "tempo", "valence", "danceability", "acousticness", "bpm"]

# Concepts that only appear in the context documents, not in songs.csv
_CONTEXT_SIGNALS = [
    "cognitive", "motor", "arousal", "parasympathetic", "concentration",
    "auditory", "timbral", "distract", "fatigue", "orient", "entrain",
    "attention", "interval", "amplitude", "dynamic", "synchron",
]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _check(response: str, expected_top: str) -> dict:
    confidence, reason = _parse_confidence(response)
    return {
        "cites_top_song":    expected_top.lower() in response.lower(),
        "cites_score":       bool(_SCORE_RE.search(response)),
        "confidence":        confidence,
        "confidence_ok":     confidence is not None and confidence >= 0.5,
        "confidence_reason": reason,
    }


def _measure_depth(response: str) -> dict:
    """Count evidence richness signals in a response."""
    text = response.lower()
    confidence, _ = _parse_confidence(response)
    return {
        "word_count":       len(response.split()),
        "attr_types_cited": sum(1 for a in _SONG_ATTRS if a in text),
        "context_signals":  sum(1 for c in _CONTEXT_SIGNALS if c in text),
        "confidence":       round(confidence, 2) if confidence is not None else 0.0,
    }


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------------------
# Eval modes
# ---------------------------------------------------------------------------

def run_eval(songs: list, api_key: str) -> None:
    """Run 4 test cases through the full multi-source pipeline."""
    assistant = RAGAssistant(csv_path=CSV_PATH, docs_paths=DOCS_PATHS, api_key=api_key)
    results = []

    for i, case in enumerate(EVAL_CASES, 1):
        print(f"\n{_DIV}")
        print(f"  Test {i}/{len(EVAL_CASES)}: {case['label']}")
        print(f"  Query        : \"{case['query']}\"")
        print(f"  Expected top : {case['expected_top']}")
        print(f"{_DIV}")
        print("  Calling Claude (blocking)...")

        recs = recommend_songs(case["user_prefs"], songs, k=5)
        response = assistant.recommend_with_scores(case["query"], recs, stream=False)
        checks = _check(response, case["expected_top"])
        results.append(checks)

        conf_str = f"{checks['confidence']:.2f}" if checks["confidence"] is not None else "N/A"
        print(f"  cites_top_song : {_status(checks['cites_top_song'])}")
        print(f"  cites_score    : {_status(checks['cites_score'])}")
        print(f"  confidence     : {conf_str}  {_status(checks['confidence_ok'])}")
        if checks["confidence_reason"]:
            print(f"  reason         : {checks['confidence_reason']}")

    n = len(results)
    top_ok   = sum(1 for r in results if r["cites_top_song"])
    score_ok = sum(1 for r in results if r["cites_score"])
    conf_vals = [r["confidence"] for r in results if r["confidence"] is not None]
    avg_conf  = sum(conf_vals) / len(conf_vals) if conf_vals else None
    all_pass  = sum(
        1 for r in results
        if r["cites_top_song"] and r["cites_score"] and r["confidence_ok"]
    )

    print(f"\n{_HDR}")
    print("  EVAL SUMMARY")
    print(_HDR)
    print(f"  Top song cited    : {top_ok}/{n}")
    print(f"  Score cited       : {score_ok}/{n}")
    print(f"  Avg confidence    : {f'{avg_conf:.2f}' if avg_conf else 'N/A'}")
    print(f"  Fully passed      : {all_pass}/{n}")
    print(_HDR)


def run_comparison(songs: list, api_key: str) -> None:
    """
    Run one query in single-source and multi-source mode, then print
    side-by-side depth metrics to quantify the improvement.
    """
    case = COMPARE_CASE
    recs = recommend_songs(case["user_prefs"], songs, k=5)

    print(f"\n{_HDR}")
    print("  COMPARISON: songs-only  vs.  multi-source")
    print(f"  Query: \"{case['query']}\"")
    print(_HDR)

    print("\n  [Mode 1 / 2]  Single-source — songs.csv only")
    assistant_s = RAGAssistant(csv_path=CSV_PATH, api_key=api_key)
    response_s  = assistant_s.recommend_with_scores(case["query"], recs, stream=False)
    metrics_s   = _measure_depth(response_s)

    print("\n  [Mode 2 / 2]  Multi-source — songs.csv + genre profiles + activity guides + artist notes")
    assistant_m = RAGAssistant(csv_path=CSV_PATH, docs_paths=DOCS_PATHS, api_key=api_key)
    response_m  = assistant_m.recommend_with_scores(case["query"], recs, stream=False)
    metrics_m   = _measure_depth(response_m)

    # Print comparison table
    print(f"\n{'Metric':<25} {'Single-source':>14} {'Multi-source':>14} {'Delta':>10}")
    print("-" * 63)
    metric_labels = {
        "word_count":       "word count",
        "attr_types_cited": "attr types cited (/8)",
        "context_signals":  "context signals",
        "confidence":       "confidence (0-1)",
    }
    for key, label in metric_labels.items():
        sv = metrics_s[key]
        mv = metrics_m[key]
        if isinstance(sv, float):
            delta_str = f"+{mv - sv:.2f}" if mv >= sv else f"{mv - sv:.2f}"
            print(f"  {label:<23} {sv:>14.2f} {mv:>14.2f} {delta_str:>10}")
        else:
            delta_str = f"+{mv - sv}" if mv >= sv else str(mv - sv)
            print(f"  {label:<23} {sv:>14} {mv:>14} {delta_str:>10}")

    print()
    gain_attrs = metrics_m["attr_types_cited"] - metrics_s["attr_types_cited"]
    gain_ctx   = metrics_m["context_signals"]  - metrics_s["context_signals"]
    gain_words = metrics_m["word_count"]       - metrics_s["word_count"]
    print(f"  Multi-source added {gain_ctx} context-specific concepts, "
          f"{gain_attrs:+d} attribute types, {gain_words:+d} words.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="VibeMatch AI evaluation harness")
    ap.add_argument(
        "--compare", action="store_true",
        help="Run single-source vs. multi-source comparison on one query"
    )
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    songs = load_songs(CSV_PATH)

    if args.compare:
        run_comparison(songs, api_key)
    else:
        run_eval(songs, api_key)


if __name__ == "__main__":
    main()
