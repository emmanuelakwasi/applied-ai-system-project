"""
VibeMatch multi-step reasoning agent.

ReasoningAssistant replaces the one-shot recommend_with_scores() call with a
tool-use agentic loop. Claude decides which tools to call, in what order, and
why. Every tool call is printed as a numbered, observable step in real time.

Five tools are available:
  score_songs          run the rule-based scorer for any user profile
  retrieve_songs       TF-IDF semantic search against the song catalog
  get_genre_profile    look up a genre's attributes and use cases
  get_activity_context look up what makes music effective for an activity
  get_artist_notes     look up notes on a specific catalog artist

Usage:
    python src/main.py "chill music for studying" --reason
"""

import json
import logging
import os
import sys
from typing import Optional

import anthropic

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)

from recommender import load_songs, recommend_songs
from rag_pipeline import SongKnowledgeBase, DocumentKnowledgeBase

_logger = logging.getLogger(__name__)

_DIV = "-" * 60
_HDR = "=" * 60
_MAX_STEPS = 12


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are VibeMatch AI, a music recommendation assistant. You have access to tools \
for gathering evidence before making a recommendation. The catalog has 10 songs \
across genres: lo-fi, ambient, rock, synthwave, pop, jazz, indie pop. Artists: \
LoRoom, Paper Lanterns, Orbit Bloom, Slow Stereo, Voltline, Max Pulse, Neon Echo.

Your process:
1. Analyse the user's request — identify the implied activity, mood, and listener type.
2. Gather evidence with tools: use score_songs to rank songs against a profile, \
retrieve_songs for semantic similarity, get_genre_profile / get_activity_context to \
understand what attributes matter, and get_artist_notes when a specific artist is \
relevant.
3. After gathering evidence, give a final recommendation that cites specific scores \
and attributes, explains WHY those attributes matter for the activity (cognitive load, \
motor synchronization, arousal, etc.), and flags any disagreement between scoring and \
semantic signals.
4. End your response with: CONFIDENCE: 0.X — one sentence on certainty.

Always call at least one tool before giving a final answer.\
"""


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "score_songs",
        "description": (
            "Run the rule-based scorer for a given user profile. Returns all 10 songs "
            "ranked by genre + mood + energy match with score breakdowns. Use this to "
            "see which songs score highest for a specific listener profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre":  {
                    "type": "string",
                    "description": "Preferred genre (e.g. 'lofi', 'rock', 'ambient', 'pop', 'jazz', 'synthwave', 'indie pop')",
                },
                "mood":   {
                    "type": "string",
                    "description": "Preferred mood (e.g. 'chill', 'intense', 'happy', 'relaxed', 'moody')",
                },
                "energy": {
                    "type": "number",
                    "description": "Target energy level 0.0 (very quiet) to 1.0 (very intense)",
                },
            },
            "required": ["genre", "mood", "energy"],
        },
    },
    {
        "name": "retrieve_songs",
        "description": (
            "Run TF-IDF semantic retrieval to find songs most similar to a query. "
            "Returns songs ranked by text similarity independent of any user profile. "
            "Use this as a second opinion alongside score_songs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g. 'late-night chill studying')",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (1-5, default 3)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_genre_profile",
        "description": (
            "Retrieve detailed profile for a music genre: typical BPM range, energy, "
            "acousticness, valence, best use cases, adjacent genres, and what makes "
            "it work or not work for specific activities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genre": {
                    "type": "string",
                    "description": "Genre name (e.g. 'lo-fi', 'ambient', 'rock', 'synthwave', 'pop', 'jazz', 'indie pop')",
                },
            },
            "required": ["genre"],
        },
    },
    {
        "name": "get_activity_context",
        "description": (
            "Retrieve research-backed guidance on what makes music effective for a "
            "specific listening activity. Covers cognitive load, motor synchronization, "
            "arousal regulation, and other physiological mechanisms."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity": {
                    "type": "string",
                    "description": "Listening context (e.g. 'studying', 'HIIT workout', 'sleep', 'morning routine', 'social background')",
                },
            },
            "required": ["activity"],
        },
    },
    {
        "name": "get_artist_notes",
        "description": (
            "Retrieve production notes for a specific catalog artist: their design "
            "intent, what distinguishes them from similar acts, and which use cases "
            "they are and are not suited for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name: LoRoom, Paper Lanterns, Orbit Bloom, Slow Stereo, Voltline, Max Pulse, or Neon Echo",
                },
            },
            "required": ["artist"],
        },
    },
]


# ---------------------------------------------------------------------------
# ReasoningAssistant
# ---------------------------------------------------------------------------

class ReasoningAssistant:
    """
    Multi-step recommendation agent with observable tool calls.

    Claude drives the evidence-gathering process: it decides which tools to
    call and in what order. Each call is printed as a numbered step with
    inputs and abbreviated output. Claude's summarized reasoning between
    steps is printed when present.
    """

    def __init__(
        self,
        csv_path: str,
        docs_paths: Optional[list[str]] = None,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
    ):
        paths = docs_paths or []
        self._songs      = load_songs(csv_path)
        self._song_kb    = SongKnowledgeBase(csv_path)
        # Separate KBs per document type for precise tool lookups
        self._genre_kb    = DocumentKnowledgeBase(
            [p for p in paths if "genre"    in os.path.basename(p)]
        )
        self._activity_kb = DocumentKnowledgeBase(
            [p for p in paths if "activity" in os.path.basename(p)]
        )
        self._artist_kb   = DocumentKnowledgeBase(
            [p for p in paths if "artist"   in os.path.basename(p)]
        )
        self.model  = model
        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recommend(self, query: str) -> str:
        """
        Run the agentic reasoning loop for a query.

        Prints each tool call and its result as a numbered step.
        Returns the final recommendation text.
        """
        print(f"\n{_HDR}")
        print("  VibeMatch Reasoning Agent")
        print(f"  Query: \"{query}\"")
        print(_HDR)

        messages = [{"role": "user", "content": f"Recommend music for: {query}"}]
        step      = 0
        final_text = ""

        while step < _MAX_STEPS:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    thinking={"type": "adaptive", "display": "summarized"},
                    system=_SYSTEM,
                    tools=_TOOLS,
                    messages=messages,
                )
            except anthropic.APIError as exc:
                _logger.error("API error in reasoning loop: %s", exc)
                raise

            # Print summarized thinking blocks when present
            for block in response.content:
                if block.type == "thinking":
                    summary = getattr(block, "thinking", "").strip()
                    if summary:
                        # Wrap at ~80 chars for readability
                        print(f"\n  [Reasoning]")
                        for line in summary.splitlines():
                            print(f"  {line}")

            if response.stop_reason == "end_turn":
                print(f"\n{_HDR}")
                print("  Final Recommendation")
                print(_HDR + "\n")
                for block in response.content:
                    if block.type == "text":
                        print(block.text)
                        final_text += block.text
                break

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        step += 1
                        result = self._dispatch(block.name, block.input, step)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return final_text

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, name: str, inputs: dict, step: int) -> str:
        """Execute a tool, print the step header + abbreviated result."""
        print(f"\n  [{step:02d}] {name}")
        print(f"       Input : {json.dumps(inputs)}")
        print(f"       {'-' * 50}")

        _fn_map = {
            "score_songs":          self._tool_score_songs,
            "retrieve_songs":       self._tool_retrieve_songs,
            "get_genre_profile":    self._tool_get_genre_profile,
            "get_activity_context": self._tool_get_activity_context,
            "get_artist_notes":     self._tool_get_artist_notes,
        }
        fn = _fn_map.get(name)
        result = fn(inputs) if fn is not None else f"Unknown tool: {name}"

        # Print up to 10 lines of the result
        lines = result.splitlines()
        for line in lines[:10]:
            print(f"       {line}")
        if len(lines) > 10:
            print(f"       ... ({len(lines) - 10} more lines)")

        return result

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_score_songs(self, inputs: dict) -> str:
        user_prefs = {
            "genre":  str(inputs["genre"]),
            "mood":   str(inputs["mood"]),
            "energy": float(inputs["energy"]),
        }
        recs = recommend_songs(user_prefs, self._songs, k=5)
        lines = [
            f"Rule-based scores  (profile: genre={inputs['genre']} "
            f"mood={inputs['mood']} energy={inputs['energy']}):"
        ]
        for i, (song, score, explanation) in enumerate(recs, 1):
            lines.append(
                f"  {i}. \"{song['title']}\" by {song['artist']}"
                f" | score {score:.2f}"
                f" | {explanation}"
                f" | valence={song['valence']} dance={song['danceability']}"
                f" acoustic={song['acousticness']} tempo={int(song['tempo_bpm'])}bpm"
            )
        return "\n".join(lines)

    def _tool_retrieve_songs(self, inputs: dict) -> str:
        k = min(int(inputs.get("k", 3)), 5)
        results = self._song_kb.retrieve(str(inputs["query"]), k=k)
        lines = [f"TF-IDF semantic matches for \"{inputs['query']}\"  (k={k}):"]
        for i, s in enumerate(results, 1):
            lines.append(
                f"  {i}. \"{s['title']}\" by {s['artist']}"
                f" | genre={s['genre']} mood={s['mood']}"
                f" | energy={s['energy']} valence={s['valence']}"
                f" dance={s['danceability']} acoustic={s['acousticness']}"
            )
        return "\n".join(lines)

    def _tool_get_genre_profile(self, inputs: dict) -> str:
        genre = str(inputs["genre"])
        results = self._genre_kb.retrieve(f"{genre} genre music profile characteristics", k=1)
        if not results:
            return f"No genre profile found for: {genre}"
        d = results[0]
        return f"[{d['title']}]\n{d['content']}"

    def _tool_get_activity_context(self, inputs: dict) -> str:
        activity = str(inputs["activity"])
        results = self._activity_kb.retrieve(
            f"{activity} music listening context effective", k=1
        )
        if not results:
            return f"No activity context found for: {activity}"
        d = results[0]
        return f"[{d['title']}]\n{d['content']}"

    def _tool_get_artist_notes(self, inputs: dict) -> str:
        artist = str(inputs["artist"])
        results = self._artist_kb.retrieve(f"{artist} artist music catalog notes", k=1)
        if not results:
            return f"No artist notes found for: {artist}"
        d = results[0]
        return f"[{d['title']}]\n{d['content']}"
