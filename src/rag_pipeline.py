"""
RAG pipeline for the VibeMatch music assistant.

Flow:
  User query
    → TF-IDF retrieval (scikit-learn) over songs.csv
    → Top-k songs injected into prompt context
    → Claude (claude-opus-4-7) generates a natural-language response
    → Streamed back to the caller

The system prompt (role + full song catalog) is marked for prompt caching so
repeated queries skip re-tokenising the stable context.
"""

import csv
import json
import logging
import os
import re
from typing import Optional

import anthropic

_logger = logging.getLogger(__name__)

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ---------------------------------------------------------------------------
# Song loading & text representation
# ---------------------------------------------------------------------------

def _load_songs(csv_path: str) -> list[dict]:
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            songs.append({
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
            })
    return songs


def _song_to_search_text(s: dict) -> str:
    """Descriptive string used to build the TF-IDF index."""
    energy_word = "high-energy" if s["energy"] > 0.65 else ("mid-energy" if s["energy"] > 0.45 else "low-energy")
    acoustic_word = "acoustic" if s["acousticness"] > 0.5 else "electronic"
    dance_word = "danceable" if s["danceability"] > 0.65 else "non-danceable"
    return (
        f"{s['title']} {s['artist']} {s['genre']} {s['mood']} "
        f"{energy_word} {acoustic_word} {dance_word} "
        f"tempo {int(s['tempo_bpm'])} bpm valence {s['valence']}"
    )


def _format_song_line(s: dict) -> str:
    return (
        f"  • \"{s['title']}\" by {s['artist']}"
        f" | genre={s['genre']} mood={s['mood']}"
        f" | energy={s['energy']} tempo={int(s['tempo_bpm'])}bpm"
        f" | valence={s['valence']} dance={s['danceability']} acoustic={s['acousticness']}"
    )


# ---------------------------------------------------------------------------
# Knowledge base — TF-IDF retrieval
# ---------------------------------------------------------------------------

class SongKnowledgeBase:
    """Lightweight TF-IDF retriever over the songs CSV."""

    def __init__(self, csv_path: str):
        if not _SKLEARN_OK:
            raise ImportError(
                "scikit-learn and numpy are required for RAG retrieval.\n"
                "Run: pip install scikit-learn numpy"
            )
        self.songs = _load_songs(csv_path)
        texts = [_song_to_search_text(s) for s in self.songs]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k songs most semantically similar to the query."""
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = scores.argsort()[::-1][:k]
        return [self.songs[i] for i in top_idx]

    @property
    def all_songs(self) -> list[dict]:
        return self.songs


# ---------------------------------------------------------------------------
# RAG assistant — Claude API + prompt caching + streaming
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Document loading & knowledge base — supplementary text retrieval
# ---------------------------------------------------------------------------

def _load_documents(paths: list[str]) -> list[dict]:
    docs = []
    for path in paths:
        if not os.path.exists(path):
            _logger.warning("Document file not found, skipping: %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for entry in json.load(f):
                docs.append(entry)
    _logger.info("Loaded %d context documents from %d path(s)", len(docs), len(paths))
    return docs


def _doc_to_search_text(d: dict) -> str:
    title = d.get("title", "")
    tags  = " ".join(d.get("tags", []))
    # Repeat title/tags to boost their TF-IDF weight relative to body text
    return f"{title} {title} {tags} {tags} {d.get('content', '')}"


def _format_doc_excerpt(d: dict, max_chars: int = 800) -> str:
    content = d.get("content", "")
    if len(content) > max_chars:
        content = content[:max_chars].rsplit(" ", 1)[0] + "..."
    return f"  [{d['title']}]\n  {content}"


class DocumentKnowledgeBase:
    """
    TF-IDF retriever over supplementary JSON documents.

    Each JSON file is a list of objects with keys:
      id, title, tags (list[str]), content (str)

    Paths that do not exist are skipped with a warning so the system
    degrades gracefully when document files are absent.
    """

    def __init__(self, paths: list[str]):
        if not _SKLEARN_OK:
            raise ImportError(
                "scikit-learn and numpy are required.\n"
                "Run: pip install scikit-learn numpy"
            )
        self.documents = _load_documents(paths)
        if not self.documents:
            self._fitted = False
            return
        self._fitted = True
        texts = [_doc_to_search_text(d) for d in self.documents]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return the top-k documents most relevant to the query."""
        if not self._fitted:
            return []
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        k = min(k, len(self.documents))
        top_idx = scores.argsort()[::-1][:k]
        return [self.documents[i] for i in top_idx]

    @property
    def count(self) -> int:
        return len(self.documents)


# ---------------------------------------------------------------------------
# RAG assistant — Claude API + prompt caching + streaming
# ---------------------------------------------------------------------------

_ROLE_DESCRIPTION = """\
You are VibeMatch AI, an expert music recommendation assistant backed by a \
curated song database and a supplementary knowledge base covering genre profiles, \
listening-activity contexts, and artist notes. Always:
- Reference specific songs by name and artist from the provided catalog.
- Justify recommendations using song attributes (genre, mood, energy, tempo, \
valence, danceability, acousticness).
- When background knowledge documents are provided, use them to explain WHY \
specific attributes matter for the user's activity — not just which numbers are \
high, but what those numbers mean for the listener's actual experience.
- Be concise: one focused paragraph per recommendation is ideal.
- For playlist debugging or classification tasks, identify patterns across songs \
(e.g. energy mismatch, genre collision, pacing issues).

You can handle four types of queries:
  recommend — suggest songs for an activity, mood, or vibe
  explain   — explain why a specific song fits a use case
  classify  — determine a song's genre, mood, or style category
  debug     — find what feels "off" in a playlist and suggest fixes\
"""


def _build_catalog_block(songs: list[dict]) -> str:
    lines = ["Full song catalog:"]
    for s in songs:
        lines.append(_format_song_line(s))
    return "\n".join(lines)


def _parse_confidence(text: str) -> tuple[float | None, str]:
    """Extract the CONFIDENCE score and reason appended by Claude to a response."""
    m = re.search(
        r"CONFIDENCE:\s*([\d.]+)\s*[—–-]\s*(.+?)(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None, ""
    try:
        return float(m.group(1)), m.group(2).strip()
    except ValueError:
        _logger.warning("Could not parse confidence value: %r", m.group(1))
        return None, ""


class RAGAssistant:
    """
    Wraps TF-IDF retrieval + Claude API.

    The stable parts (role + full catalog) are placed in the system prompt
    with cache_control so repeated queries benefit from prompt caching.

    Two main entry points:
      ask()                  — open-ended Q&A (explain, classify, debug)
      recommend_with_scores() — integrated mode: rule-based scores + TF-IDF
                                context handed to Claude as structured evidence
    """

    def __init__(
        self,
        csv_path: str,
        docs_paths: Optional[list[str]] = None,
        model: str = "claude-opus-4-7",
        api_key: Optional[str] = None,
    ):
        self.kb = SongKnowledgeBase(csv_path)
        self.doc_kb = DocumentKnowledgeBase(docs_paths or [])
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        # Pre-build the cached system prompt (stable across all queries)
        self._system = [
            {
                "type": "text",
                "text": f"{_ROLE_DESCRIPTION}\n\n{_build_catalog_block(self.kb.all_songs)}",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def recommend_with_scores(
        self,
        query: str,
        scored_songs: list[tuple],
        stream: bool = True,
    ) -> str:
        """
        Core integration method: Claude uses rule-based scores + TF-IDF
        retrieval together as evidence to formulate a recommendation.

        Args:
            query:        User's natural language request.
            scored_songs: Output of recommend_songs() — list of
                          (song_dict, score, explanation) tuples, best first.
            stream:       Stream the response to stdout while returning it.

        The prompt gives Claude three lenses on the same data:
          1. Rule-based rank   — how well each song matched genre/mood/energy
          2. Score breakdown   — which scoring components fired and why
          3. TF-IDF rank       — semantic similarity to the raw query text

        Claude is asked to cross-reference all three, validate the top picks,
        surface songs the scorer undervalued, and flag attribute gaps the
        numerical system couldn't capture.
        """
        # Signal 1: semantic retrieval (independent of rule-based profile)
        retrieved = self.kb.retrieve(query, k=3)
        retrieved_titles = {s["title"] for s in retrieved}

        # Format the scored list with a semantic-match flag
        score_lines = []
        for rank, (song, score, explanation) in enumerate(scored_songs, 1):
            sem_flag = "  <- also top semantic match" if song["title"] in retrieved_titles else ""
            score_lines.append(
                f"  {rank}. \"{song['title']}\" by {song['artist']}"
                f" | rule-based score: {score:.2f}"
                f" | {explanation}"
                f" | valence={song['valence']} dance={song['danceability']}"
                f" acoustic={song['acousticness']} tempo={int(song['tempo_bpm'])}bpm"
                f"{sem_flag}"
            )

        sem_lines = []
        for rank, s in enumerate(retrieved, 1):
            sem_lines.append(
                f"  {rank}. \"{s['title']}\" by {s['artist']}"
                f" | {s['genre']} / {s['mood']}"
                f" | energy={s['energy']} valence={s['valence']}"
                f" dance={s['danceability']} acoustic={s['acousticness']}"
            )

        # Signal 2: context documents (genre profiles, activity guides, artist notes)
        context_docs = self.doc_kb.retrieve(query, k=3)
        if context_docs:
            ctx_block = (
                "\n\n--- Background knowledge (genre profiles and activity context) ---\n"
                + "\n\n".join(_format_doc_excerpt(d) for d in context_docs)
            )
            context_instruction = (
                "5. Draw on the background knowledge section above to explain WHY the "
                "relevant attributes matter for this specific activity — not just that "
                "a song scores well, but what those attribute values mean for the "
                "listener's actual experience (cognitive load, motor synchronization, "
                "arousal level, etc.).\n"
            )
            conf_num = "6"
        else:
            ctx_block = ""
            context_instruction = ""
            conf_num = "5"

        user_message = (
            f"User request: {query}\n\n"
            f"--- Rule-based scoring output (ranked by genre + mood + energy match) ---\n"
            + "\n".join(score_lines)
            + "\n\n--- Semantic retrieval output (TF-IDF cosine similarity to the query) ---\n"
            + "\n".join(sem_lines)
            + ctx_block
            + "\n\nUsing all evidence above, give a final recommendation. "
            "Your response must:\n"
            "1. State which song(s) you recommend and why, citing specific scores and attributes.\n"
            "2. Note any cases where the rule-based ranking and semantic search disagree — "
            "and explain which source is more trustworthy for this request.\n"
            "3. Point out any attributes the scoring formula ignored "
            "(e.g. tempo, valence, acousticness, danceability) that are actually relevant.\n"
            "4. Flag any top-scored song that might not actually fit the request despite its high score.\n"
            + context_instruction
            + f"{conf_num}. End your response with a confidence score on the last line using exactly this format:\n"
            "   CONFIDENCE: 0.X — one sentence on what you are or are not certain about."
        )

        if stream:
            return self._stream(user_message)
        return self._blocking(user_message)

    def ask(self, query: str, k: int = 3, stream: bool = True) -> str:
        """
        Open-ended Q&A: retrieve k songs via TF-IDF, then let Claude answer.
        Use this for explain / classify / debug queries.
        """
        retrieved = self.kb.retrieve(query, k=k)
        context_lines = ["Most relevant songs for this query:"]
        for s in retrieved:
            context_lines.append(_format_song_line(s))

        context_docs = self.doc_kb.retrieve(query, k=2)
        if context_docs:
            context_lines.append("\nBackground knowledge:")
            for d in context_docs:
                context_lines.append(_format_doc_excerpt(d, max_chars=400))

        user_message = f"Query: {query}\n\n" + "\n".join(context_lines)

        if stream:
            return self._stream(user_message)
        return self._blocking(user_message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stream(self, user_message: str) -> str:
        full_text = ""
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=self._system,
                messages=[{"role": "user", "content": user_message}],
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    full_text += text
            print()
        except anthropic.APIError as exc:
            _logger.error("Claude API error during streaming: %s", exc)
            raise
        return full_text

    def _blocking(self, user_message: str) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=self._system,
                messages=[{"role": "user", "content": user_message}],
            )
            return next(b.text for b in response.content if b.type == "text")
        except anthropic.APIError as exc:
            _logger.error("Claude API error during blocking call: %s", exc)
            raise
