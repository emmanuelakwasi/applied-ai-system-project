# VibeMatch — Rule-Based Music Recommender + RAG AI Synthesis

A music recommendation system that combines transparent, explainable scoring with
Claude-powered analysis. Built as a two-stage pipeline: a deterministic rule-based
engine first ranks songs, then a Retrieval-Augmented Generation (RAG) layer passes
those ranked results to Claude, which cross-references them against semantic search
results and reasons over both to produce a nuanced, cited recommendation.

---

## Original Project: VibeMatch 1.0 (Modules 1-3)

**VibeMatch 1.0** was built as a transparent, classroom-scale music recommender.
Its goal was to show how a small set of explicit rules — genre match, mood match, and
energy similarity — could turn structured song metadata into ranked recommendations
without any machine learning. The system scored each song in a 10-song CSV catalog
against a user profile and returned the top results with plain-language explanations
of every scoring decision, making the algorithm fully auditable at every step.

---

## What This Project Does and Why It Matters

Most music recommenders are black boxes. VibeMatch 2.0 deliberately keeps the scoring
transparent and then adds an AI layer that must engage with those scores rather than
replace them. Claude receives both the rule-based rankings and an independent TF-IDF
semantic retrieval result, then has to reconcile disagreements between the two signals,
name attributes the formula ignored, and justify its final recommendation with specific
evidence from the data.

The result is a system that is simultaneously auditable (you can run `--classic` to
see exactly what the formula decided) and intelligent (the AI catches cases where the
formula gets the right number for the wrong reason). This architecture reflects a
real pattern in production AI: use deterministic systems for speed and interpretability,
use LLMs for nuance and synthesis.

---

## Architecture Overview

```
User query (natural language)
    |
    +-- Profile Inferrer ---------> UserProfile (genre / mood / energy)
    |                                      |
    |                                      v
    +-- SongKnowledgeBase          Rule-Based Scorer
    |   TF-IDF over songs.csv        (recommender.py)
    |        |                            |
    |        | top-k semantic matches     | ranked songs + score breakdowns
    |        +-------------+  +----------+
    |                      |  |
    +-- DocumentKnowledgeBase         |
        TF-IDF over 3 document sets:  |
          genre_profiles.json         |
          activity_contexts.json      |
          artist_notes.json           |
              |                       |
              | top-k context docs    |
              +----------+  +---------+
                         |  |
                         v  v
                   RAGAssistant
                recommend_with_scores()
                         |
                structured prompt with:
                - rule-based rank + score breakdowns  (Signal 1)
                - semantic song matches               (Signal 2)
                - genre / activity context docs       (Signal 3)
                         |
                         v
                Claude (claude-opus-4-7)
                adaptive thinking + streaming
                prompt cache on stable context
                         |
                         v
                Streamed recommendation
                citing scores, context knowledge,
                flagging formula gaps
```

Full diagram with component descriptions: [assets/system_diagram.md](assets/system_diagram.md)

The system has two independently testable layers. The rule-based scorer and the TF-IDF
retriever each run without an API key and are covered by `pytest` suites. The AI synthesis
layer sits on top and is only invoked when `ANTHROPIC_API_KEY` is set. A `--classic`
flag skips the AI entirely, letting you compare the formula's output against Claude's
interpretation side by side. A `--reason` flag switches to a full tool-use agentic loop
where Claude drives its own evidence-gathering and every intermediate step is visible.

---

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- An Anthropic API key (only required for AI synthesis; rule-based mode works without one)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd applied-ai-system-project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `anthropic`, `scikit-learn`, `numpy`, `pandas`, `pytest`, `streamlit`

### 4. Set your API key (for AI synthesis only)

```bash
# macOS / Linux
export ANTHROPIC_API_KEY=your_key_here

# Windows Command Prompt
set ANTHROPIC_API_KEY=your_key_here

# Windows PowerShell
$env:ANTHROPIC_API_KEY="your_key_here"
```

### 5. Run the recommender

```bash
# Default profile — High-Energy Pop, with AI synthesis if key is set
python src/main.py

# Natural language query
python src/main.py "chill music for late-night studying"

# Preset profile
python src/main.py --profile "Chill Lofi"

# Rule-based scores only, no AI (no API key needed)
python src/main.py "workout music" --classic

# Multi-step reasoning agent — Claude calls tools; every step is visible
python src/main.py "chill music for late-night studying" --reason

# Interactive Q&A mode (explain, classify, debug)
python src/rag_cli.py

# Evaluation harness (quality checks + single vs. multi-source comparison)
python src/eval.py
python src/eval.py --compare
```

### 6. Run the tests

```bash
pytest tests/ -v
```

All 58 tests cover the scoring, retrieval, document KB, confidence parsing, and
reasoning tool layers and pass without an API key.

---

## Sample Interactions

### Interaction 1: Classic mode — rule-based scores only

This shows the deterministic layer running standalone. No AI is involved.

**Input:**
```bash
python src/main.py "chill music for late-night studying" --classic
```

**Output:**
```
----------------------------------------------------------
  Step 1 | Rule-Based Scores  ["chill music for late-night studying"]
----------------------------------------------------------
  1. Midnight Coding by LoRoom  (score 5.92)
     Strong fit because of genre match (+1.0), mood match (+1.0), energy closeness (+3.92).
  2. Library Rain by Paper Lanterns  (score 5.80)
     Strong fit because of genre match (+1.0), mood match (+1.0), energy closeness (+3.80).
  3. Focus Flow by LoRoom  (score 5.00)
     Strong fit because of genre match (+1.0), energy closeness (+4.00).
  4. Spacewalk Thoughts by Orbit Bloom  (score 4.52)
     Strong fit because of mood match (+1.0), energy closeness (+3.52).
  5. Coffee Shop Stories by Slow Stereo  (score 3.88)
     Strong fit because of energy closeness (+3.88).
```

---

### Interaction 2: Full pipeline — AI synthesis over rule-based scores

**Input:**
```bash
python src/main.py "chill music for late-night studying"
```

**Output (Step 1 is identical to above, then Step 2 streams):**
```
----------------------------------------------------------
  Step 2 | AI Synthesis  [Claude reasoning over retrieved evidence]
----------------------------------------------------------

The rule-based scorer and semantic retrieval agree strongly here: Midnight Coding by
LoRoom is the clear top pick. It earns the highest rule-based score (5.92) through
genre, mood, and near-perfect energy alignment, and it also surfaces as the top
semantic match for "late-night studying" — both signals point to the same song, which
is a reliable indicator of a genuine fit.

Library Rain by Paper Lanterns is a confident second choice for the same reasons:
it matches lo-fi/chill/low-energy and scores 5.80, with its high acousticness (0.86)
making it particularly well-suited for a quiet, focused late-night environment — an
attribute the scoring formula did not reward but which matters for this request.

One notable flag: Focus Flow (score 5.00) ranked third by the formula on genre and
energy alone, but its mood is "focused" rather than "chill." For studying it is
probably a better fit than the mood label suggests — the semantic retriever did rank
it third — but be aware the mood mismatch means the formula is rewarding it for
different reasons than the query implies.

Spacewalk Thoughts by Orbit Bloom deserves more credit than its score of 4.52
implies. Its ambient genre, very low energy (0.28), and the highest acousticness in
the catalog (0.92) make it an excellent late-night background track. The formula
penalised it for lacking a genre match, but for this particular request, ambient is
arguably a better fit than lo-fi.

Avoid Coffee Shop Stories (score 3.88) for late-night studying — its jazz tempo
(90 BPM) and "relaxed" rather than "focused" character suits a coffee shop context,
not sustained concentration.
```

---

### Interaction 3: Intense workout query — where the formula and semantics diverge

**Input:**
```bash
python src/main.py "high-intensity interval training playlist"
```

**Output (Step 1):**
```
----------------------------------------------------------
  Step 1 | Rule-Based Scores  ["high-intensity interval training playlist"]
----------------------------------------------------------
  1. Storm Runner by Voltline  (score 5.96)
     Strong fit because of genre match (+1.0), mood match (+1.0), energy closeness (+3.96).
  2. Gym Hero by Max Pulse  (score 4.88)
     Strong fit because of mood match (+1.0), energy closeness (+3.88).
  3. Sunrise City by Neon Echo  (score 3.68)
     Strong fit because of energy closeness (+3.68).
  ...
```

**Output (Step 2 AI synthesis):**
```
Storm Runner by Voltline and Gym Hero by Max Pulse are the right picks here, and
the two evidence sources agree. Storm Runner scores highest (5.96) with full genre,
mood, and energy alignment; Gym Hero is close behind at 4.88 with near-maximum
energy (0.93) and an intense mood — its pop genre is different from rock, but for
HIIT training that distinction matters less than the energy level.

The formula's third pick, Sunrise City by Neon Echo (3.68), is worth questioning
for this request. Its high energy (0.82) earns it points, but its mood is "happy"
rather than "intense" and its danceability (0.79) and valence (0.84) describe an
upbeat pop track rather than a driving workout song. It may fit a warm-up set but
is probably misplaced in the middle of an interval session.

Night Drive Loop by Neon Echo (score 3.40) appeared in the semantic retrieval
results despite its lower rule-based rank. For HIIT, its synthwave genre and moody
character make it a reasonable inclusion for high-focus intervals, even though the
formula only rewards its energy proximity. Danceability (0.73) and a faster tempo
than its energy level suggests make it a candidate worth testing.
```

---

### Interaction 4: Interactive Q&A mode — explain a specific song

```bash
python src/rag_cli.py
```

**Input:**
```
You: Why does "Spacewalk Thoughts" keep appearing in my chill recommendations even 
though it scores lower than the lo-fi songs?
```

**Output (streamed):**
```
"Spacewalk Thoughts" by Orbit Bloom consistently appears in chill recommendations
because its attributes — energy 0.28, acousticness 0.92, tempo 60 BPM, and an
ambient genre — collectively describe a very low-stimulation listening environment.
The rule-based scorer penalizes it whenever your profile specifies lo-fi as a
favorite genre, because ambient does not match lo-fi as an exact string. That is a
known limitation of exact-match genre scoring.

What the formula misses is that ambient is, in most listening contexts, functionally
interchangeable with lo-fi for chill or focus use cases. Spotify and similar platforms
routinely cross-recommend across these two genres for "study" and "sleep" playlists.
If you wanted to fix this in the scoring system, you could group genres into
super-categories (lo-fi, ambient, and jazz all contributing to a "low-energy" bucket)
and award partial match points rather than requiring an exact string match.
```

---

### Interaction 5: Multi-step reasoning agent — observable tool calls

The `--reason` flag switches to an agentic loop. Claude decides what to look up and
every tool call is printed as a numbered step with its inputs and abbreviated output.
Claude's summarized thinking between steps is shown as `[Reasoning]` blocks.

**Input:**
```bash
python src/main.py "chill music for late-night studying" --reason
```

**Output:**
```
============================================================
  VibeMatch Reasoning Agent
  Query: "chill music for late-night studying"
============================================================

  [Reasoning]
  The request is for late-night studying — a focus/concentration use case
  with a chill, low-energy preference. I'll start by scoring songs against
  a lo-fi/chill profile, then look up the study activity context to understand
  which attributes actually matter physiologically.

  [01] score_songs
       Input : {"genre": "lofi", "mood": "chill", "energy": 0.4}
       --------------------------------------------------
       Rule-based scores  (profile: genre=lofi mood=chill energy=0.4):
         1. "Midnight Coding" by LoRoom | score 5.92 | genre match (+1.0),
            mood match (+1.0), energy closeness (+3.92) | valence=0.41
            dance=0.52 acoustic=0.82 tempo=72bpm
         2. "Library Rain" by Paper Lanterns | score 5.80 | ...
       ... (4 more lines)

  [Reasoning]
  Midnight Coding and Library Rain score clearly highest. Before finalizing,
  I want to check what the study activity context says about which attributes
  matter most — is high acousticness actually important, or is it just energy?

  [02] get_activity_context
       Input : {"activity": "late-night studying"}
       --------------------------------------------------
       [Music for Deep Focus and Studying]
       Effective study music minimizes cognitive interference. The primary
       constraint is language processing: lyrics compete directly with reading...
       ... (18 more lines)

  [Reasoning]
  The context confirms: no lyrics, consistent energy (no sudden changes),
  high acousticness reduces ear fatigue. This explains why Midnight Coding
  scores so well — its acoustic=0.82 and tempo=72bpm are both ideal.
  Let me also check semantic retrieval as a second opinion.

  [03] retrieve_songs
       Input : {"query": "late-night chill studying lo-fi", "k": 3}
       --------------------------------------------------
       TF-IDF semantic matches (k=3):
         1. "Midnight Coding" by LoRoom | genre=lofi mood=chill | energy=0.35 ...
         2. "Library Rain" by Paper Lanterns | genre=lofi mood=chill | ...
         3. "Focus Flow" by LoRoom | genre=lofi mood=focused | ...

  [04] get_artist_notes
       Input : {"artist": "LoRoom"}
       --------------------------------------------------
       [LoRoom: Artist Notes]
       LoRoom is a bedroom producer whose catalog is designed explicitly for
       long-duration study sessions. Production signature: heavily tape-saturated
       drums tuned to 65-80 BPM, Rhodes electric piano samples with vinyl crackle...
       ... (6 more lines)

============================================================
  Final Recommendation
============================================================

Midnight Coding by LoRoom is the clear top recommendation for late-night
studying. It earns the highest rule-based score (5.92) through genre, mood,
and near-perfect energy alignment, and it also tops the semantic retrieval
results — both signals agree, which is a reliable indicator of genuine fit.

The activity context explains why the attributes matter here: the absence of
lyrics eliminates the language-processing competition that degrades reading
comprehension; the tempo of 72 BPM is below the threshold (90 BPM) where
motor arousal disrupts focus; and the high acousticness (0.82) reduces the
sharp synthetic harmonics that cause auditory fatigue over a long session.
LoRoom's design intent — loop-based structures with no melodic climax or
dynamic peak — is exactly what the study context requires.

Library Rain by Paper Lanterns is an equally strong second choice with an
even higher acousticness (0.85), making it particularly well-suited for
sessions where any rhythmic element feels distracting. The artist notes
confirm it was designed for reading in quiet environments.

Focus Flow (score 5.00) is a reasonable third, though its mood is "focused"
rather than "chill" — the semantic retriever correctly ranks it third rather
than second, suggesting it is a slightly different emotional register from
the "late-night" implied tone of the request.

CONFIDENCE: 0.93 — both scoring and semantic signals agree on the top two picks,
and the study activity context confirms all three relevant attributes.
```

The key difference from the standard mode: in `--reason` mode, Claude is not handed
pre-computed results. It decides to call `score_songs` before anything else, then
decides it needs `get_activity_context` to understand the criteria, then validates
with `retrieve_songs`, and finally checks `get_artist_notes` because the same artist
appeared twice. Each of those decisions is visible as it happens, and the final answer
is grounded in everything Claude chose to look up rather than everything pre-fed to it.

---

## Design Decisions

### Why TF-IDF rather than an embedding model?

TF-IDF runs entirely offline with no API calls and no model downloads. For a 10-song
catalog it is more than sufficient, and it keeps the retrieval layer independently
testable without mocking an external service. The bigram configuration
(`ngram_range=(1,2)`) captures phrases like "high energy" and "chill lo-fi" that
single-word tokenization would split apart. If the catalog grew to thousands of songs,
a sentence-transformer or Anthropic embedding model would be the right replacement.

### Why keep the rule-based scorer rather than replacing it with Claude?

The scorer provides two things Claude cannot: deterministic, auditable decisions and
zero API cost. It also creates a concrete disagreement surface — when the formula ranks
a song highly but Claude flags it as a mismatch, that is a meaningful signal about
the formula's limitations, not just AI opinion. Replacing the scorer entirely would
lose the interpretability that makes this system trustworthy.

### Why pass scores to Claude rather than just song metadata?

The original standalone RAG script retrieved songs and handed them to Claude as a list.
Claude could then say anything, anchored only by the song attributes. By passing the
pre-computed scores and breakdowns as part of the prompt, Claude is forced to engage
with the specific reasoning the formula used — agreeing, challenging, or extending it —
rather than generating a recommendation from scratch. This is the distinction between
decoration and genuine augmentation.

### Why prompt caching on the system prompt?

The system prompt contains the role description and the full song catalog, which is
identical across every query. Marking it with `cache_control: ephemeral` means the
first request pays the tokenization cost once; every subsequent query reads that prefix
from cache at roughly 10% of the input token price. On a 10-song catalog this saving
is modest, but the pattern is directly applicable to larger catalogs where the system
prompt could be tens of thousands of tokens.

### Trade-offs made

| Decision | Benefit | Cost |
|---|---|---|
| TF-IDF for retrieval | Zero latency, no API calls, testable offline | No semantic understanding; "rock" and "intense" are unrelated tokens |
| Keyword heuristic for profile inference | Simple, transparent, no AI needed | Misclassifies ambiguous queries; "indie morning run" would default to pop |
| Exact-match genre/mood scoring | Fully explainable, easy to audit | Cannot reward near-miss genres (ambient vs. lo-fi) |
| Streaming responses | User sees output immediately; avoids HTTP timeouts | Makes unit testing the AI layer harder without mocking |
| `--classic` fallback | System usable without an API key | Creates two code paths that must stay in sync |
| `--reason` agentic loop | Reasoning chain is fully observable; Claude adapts its research strategy per query | One extra API round-trip per tool call; slower than one-shot synthesis |

---

## Testing Summary

Three complementary layers verify that the system works rather than merely appears to:
automated unit tests cover the deterministic components; an AI evaluation harness
checks the quality of Claude's synthesis on four known queries; and confidence scoring
gives every response a structured confidence score the eval harness can read automatically.

**33/33 unit tests pass without an API key. Eval harness: 4/4 queries cited the
expected top-scored song by name; 4/4 responses contained score citations; average
confidence 0.84. The `--compare` mode shows multi-source retrieval adds an average
of 4+ context-specific concepts per response that are entirely absent in single-source
mode — the measurable improvement from adding genre profiles and activity guides.**

### What the tests cover

**`tests/test_recommender.py`** (2 tests)
- Verifies that the scored list is sorted from highest to lowest score
- Verifies that `explain_recommendation` returns a non-empty string

**`tests/test_rag_pipeline.py`** (20 tests)
- `SongKnowledgeBase` loads all 10 songs and returns the correct number of results
- The `retrieve()` method returns `k` results for any `k` up to the catalog size
- Genre-specific queries surface the expected genres in the top-3 (lo-fi for chill queries, intense for workout queries)
- Acoustic queries return at least one high-`acousticness` song
- All returned `dicts` contain the required keys with correct types
- Helper functions produce strings that contain the expected song metadata
- `DocumentKnowledgeBase` loads 0, 1, or many documents from JSON files
- Missing file paths are skipped gracefully without raising exceptions
- Retrieve returns the most relevant document for a genre or activity query
- `k` parameter is respected; real data files (`genre_profiles.json`, `activity_contexts.json`) load correctly

**`tests/test_ai_eval.py`** (11 tests — no API key needed)
- `_parse_confidence()` correctly extracts score and reason from long-dash and hyphen separator formats
- Returns `None` gracefully for missing or malformed confidence lines
- Matching is case-insensitive and works mid-text
- Quality-check helpers correctly detect presence/absence of song name and score citations

All 33 tests pass without an API key:

```
tests/test_ai_eval.py::TestParseConfidence::test_valid_em_dash           PASSED
tests/test_ai_eval.py::TestParseConfidence::test_valid_hyphen             PASSED
tests/test_ai_eval.py::TestParseConfidence::test_missing_returns_none     PASSED
tests/test_ai_eval.py::TestParseConfidence::test_malformed_no_separator   PASSED
tests/test_ai_eval.py::TestParseConfidence::test_case_insensitive         PASSED
tests/test_ai_eval.py::TestParseConfidence::test_mid_text_confidence      PASSED
tests/test_ai_eval.py::TestQualityChecks::test_cites_top_song_present     PASSED
... (22 more passing)
33 passed in 9.43s
```

### AI Evaluation Harness

`python src/eval.py` runs four integration test cases through the full multi-source
pipeline and checks three quality criteria per response. `python src/eval.py --compare`
runs one query in both single-source (songs.csv only) and multi-source mode and prints
depth metrics side by side to quantify the improvement numerically.

```
python src/eval.py

  Test 1/4: chill/lo-fi query — Expected top: Midnight Coding
  cites_top_song : PASS  |  cites_score: PASS  |  confidence: 0.91  PASS

  Test 3/4: ambient/sleep query — Expected top: Spacewalk Thoughts
  cites_top_song : PASS  |  cites_score: PASS  |  confidence: 0.72  PASS
  reason: rule-based and semantic signals partially disagree on ambient vs. lo-fi

  EVAL SUMMARY
    Top song cited : 4/4  |  Score cited: 4/4  |  Avg confidence: 0.84  |  Fully passed: 4/4
```

```
python src/eval.py --compare

  COMPARISON: songs-only  vs.  multi-source
  Query: "chill music for late-night studying"

  Metric                    Single-source   Multi-source      Delta
  ---------------------------------------------------------------
  word count                          187            241        +54
  attr types cited (/8)                 4              6         +2
  context signals                       0              5         +5
  confidence (0-1)                   0.82           0.91      +0.09

  Multi-source added 5 context-specific concepts, +2 attribute types, +54 words.
```

The `context signals` metric counts mentions of concepts (cognitive load, motor
synchronization, arousal, entrainment, etc.) that exist only in the context documents,
not in the song CSV. A single-source response always scores 0; the multi-source
response draws from the activity context guide and genre profile to explain the
neurological and physiological reasons why specific attributes matter for the use case
— not just that a song scores well, but what that score means for the listener.

The lower confidence on the ambient query (0.72) reflects a genuine limitation: the
rule-based scorer penalizes "ambient" for not matching "lo-fi" exactly, while the
semantic retriever ranks it first. Claude's self-assessment correctly flags the
disagreement rather than masking it.

### Error Logging

`rag_pipeline.py` uses Python's `logging` module. Any `anthropic.APIError` — network
failures, rate limits, invalid keys — is logged at `ERROR` level before re-raising,
so the call stack and error type are preserved for debugging rather than silently
swallowed. To enable log output:

```python
import logging
logging.basicConfig(level=logging.INFO)  # or ERROR for quiet production use
```

### What worked well

The TF-IDF retriever consistently returned semantically appropriate songs for genre
and mood queries. The genre-specific tests pass reliably because the song descriptions
include genre, mood, and energy labels as searchable tokens. The rule-based scorer
behaved exactly as specified: deterministic, auditable, and easy to reason about.

The confidence scoring addition turned out to be a useful signal in itself: on queries
where both evidence sources agree, Claude's confidence was consistently above 0.85; on
queries where they diverge, it dropped toward 0.70. That correlation suggests the
self-assessment is tracking real signal rather than always returning a high number.

### What did not work as expected

The profile inference layer uses simple keyword matching and fails on queries that do not
contain the mapped keywords. "Something for getting into flow state" would fall through
to the default High-Energy Pop profile rather than a lo-fi/focused profile. A future
version should either use Claude to infer a profile from the query or support a
broader synonym list.

The `--classic` mode exposed a scoring blind spot: the energy formula assigns identical
scores to songs that are equidistant from the target energy regardless of direction.
A song that is slightly too intense and a song that is slightly too mellow receive the
same points, even though a user often has a directional preference.

### What was learned

Writing tests for the retrieval layer before integrating Claude was the most valuable
decision in this project. Because the `SongKnowledgeBase` is independently testable,
it was possible to verify that the retriever was working correctly before involving the
API at all. This mirrors how production AI systems should be structured: every
deterministic component should be testable in isolation so that failures can be
localized quickly.

The eval harness reinforced a practical point: self-reported confidence is only useful
if it varies. A model that always returns 0.9 tells you nothing. By checking whether
the confidence scores correlate with the actual signal-agreement level between the two
retrieval sources, the harness confirmed that Claude's self-assessments were anchored
in the data rather than being formulaic.

---

## Responsible AI: Limitations, Bias, and Collaboration

### Limitations and biases in this system

The most significant bias is structural: every scoring and retrieval decision is built
on a 10-song catalog that contains only Western, anglophone genres. A user querying for
Afrobeat, K-pop, or cumbia will get results mapped to the nearest Western proxy. That
is not a bug that can be patched without replacing the catalog — it is a data bias baked
into the foundation.

Within the catalog, three narrower limitations matter:

- **Exact-match genre scoring** treats "ambient" and "lo-fi" as completely unrelated,
  even though they serve the same listening contexts. Any song outside the exact string
  that matches a user's genre preference is silently penalized, which systematically
  undervalues adjacent genres.
- **Keyword-to-profile mapping** encodes cultural assumptions. Mapping "gym" to
  rock/intense is a reasonable heuristic for one cultural context and wrong for many
  others. Users whose mental model of workout music is EDM, hip-hop, or cumbia will
  receive profiles that do not reflect their intent.
- **Energy directionality is ignored.** The formula scores a song that is slightly too
  loud the same as one that is slightly too quiet, even though listener preference is
  usually directional ("at least this energetic" differs from "at most this energetic").

### Could this AI be misused, and how would it be prevented?

At music-recommendation scale, the direct harm potential is low. But the underlying
architecture — deterministic scoring feeding into LLM synthesis with confidence
signals — is a general pattern, and that pattern can be misused when deployed in
higher-stakes domains.

Two specific risks worth naming: first, the confidence scoring could be tuned to
output artificially high values, creating a false sense of reliability in a system
that is actually uncertain. A user who sees "CONFIDENCE: 0.95" on every response
learns nothing from the signal. Second, the "explain/classify/debug" mode could be
used to generate authoritative-sounding descriptions of songs at scale — useful for
content farms or manufactured playlists that want to appear curated.

The mitigations built into this system are modest but intentional: the `--classic`
flag exposes the raw scores so a user can always check what the formula actually
decided; the prompt requires Claude to flag disagreements rather than paper over them;
and the eval harness catches cases where confidence scores stop varying (a sign that
the model is no longer anchored in the data). For a production deployment, additional
guardrails — rate limits, output logging, human review of flagged responses — would
be necessary.

### What surprised me during reliability testing

The most unexpected finding was that Claude's self-reported confidence scores varied
in a meaningful, interpretable way rather than clustering near 1.0 as a default. The
ambient-sleep test case consistently produced confidence around 0.72, lower than the
other three cases. That number was not arbitrary: it corresponded exactly to the query
where the rule-based scorer and TF-IDF retriever disagreed most — the rule-based
scorer penalized "Spacewalk Thoughts" for its ambient genre not matching lo-fi, while
the semantic retriever ranked it first. Claude flagged the same disagreement in prose
and expressed lower confidence in its recommendation. The two signals — the score and
the text — were saying the same thing independently. That correlation was not designed
in; it emerged from the prompt structure.

The second surprise was how often the formula and semantic retriever diverged. Going
in, the assumption was that the two signals would mostly agree and Claude would mainly
confirm the top pick. In practice, roughly one in three test queries produced a
meaningful rank difference between the two sources — and those were always the most
informative responses, because Claude had something real to reconcile rather than just
paraphrase the scores.

### Collaboration with AI during this project

**One instance where the AI's suggestion was genuinely helpful:** When asked to make
the RAG integration "deep" rather than decorative, the suggestion was to pass the
pre-computed rule-based scores directly into the prompt as structured evidence and
require Claude to cross-reference them against the semantic retrieval results, with
explicit instructions to cite score values, flag disagreements, and name attributes
the formula ignored. That prompt structure — handing Claude two independent views of
the same data and asking it to reconcile them — was the decision that changed the
quality of the system's output. The AI went from generating generic music commentary
to producing analysis that was grounded in specific numbers and acknowledged the
scoring system's limits.

**One instance where the AI's suggestion was flawed:** During development, the initial
`main.py` used Unicode box-drawing characters (`─` U+2500) as visual dividers in the
terminal output. The AI did not flag any issue with this choice — and on Windows, the
default terminal code page (cp1252) cannot encode that character, so every run crashed
with a `UnicodeEncodeError` before printing a single line of output. The fix was
trivial (replacing `─` with `-`), but the failure itself was a useful reminder: AI
tools trained primarily on Unix-centric codebases will default to patterns that break
silently on Windows. Platform-specific encoding constraints are exactly the kind of
environmental detail that gets missed when suggestions are generated without running
in the target environment.

---

## Reflection

Building this project clarified something that is easy to miss when working with AI
tools: the most important architectural decision is not which model to use, but what
role the model should play. In the first version of the standalone RAG script, Claude
was handed a list of songs and asked to comment on them. That is a decoration pattern —
the AI adds prose around data it did not participate in retrieving or evaluating. The
integration step, passing the rule-based scores to Claude as structured evidence and
requiring it to cross-reference them against the semantic retrieval results, turned
decoration into reasoning. Claude's output changed from generic music commentary to
specific, data-grounded analysis that cited score breakdowns and flagged formula
failures. The architecture determined the quality of the AI's contribution, not the
model itself.

The second thing this project reinforced is how much transparency matters for trust.
Because the rule-based scorer explains every point it awards, a user can look at
VibeMatch's recommendation and verify the reasoning themselves. When Claude then
challenges or extends that reasoning, the disagreement is legible — you can see exactly
what the formula said and exactly what Claude said differently. That kind of
interpretability is not just academically nice; it is the difference between a system
you can debug and a system you can only hope works. As AI becomes more embedded in
decisions that affect people, the ability to show your work is going to matter as much
as accuracy.

---

## Project Structure

```
applied-ai-system-project/
├── src/
│   ├── main.py              # Entry point: --classic / default RAG / --reason modes
│   ├── recommender.py       # Rule-based scorer, Song/UserProfile dataclasses
│   ├── rag_pipeline.py      # SongKnowledgeBase + DocumentKnowledgeBase + RAGAssistant
│   ├── reasoning.py         # ReasoningAssistant: tool-use agentic loop with observable steps
│   ├── rag_cli.py           # Interactive Q&A CLI (explain, classify, debug)
│   ├── eval.py              # Eval harness: quality checks + single vs. multi-source compare
│   └── run_edge_cases.py    # Edge case profiles from Module 3
├── tests/
│   ├── test_recommender.py  # Scorer tests (sort order, explanations) — 2 tests
│   ├── test_rag_pipeline.py # Retrieval tests: SongKB, DocumentKB, helpers — 20 tests
│   ├── test_ai_eval.py      # Confidence parsing + quality check unit tests — 11 tests
│   └── test_reasoning.py    # Tool implementation unit tests (all 5 tools) — 25 tests
├── data/
│   ├── songs.csv            # 10-song catalog, 9 attributes each
│   ├── genre_profiles.json  # 7 genre documents (lo-fi, ambient, rock, synthwave, pop, jazz, indie)
│   ├── activity_contexts.json  # 5 activity documents (study, HIIT, sleep, morning, social)
│   └── artist_notes.json    # 7 artist documents (one per artist in the catalog)
├── assets/
│   └── system_diagram.md    # Mermaid architecture diagram
├── model_card.md            # VibeMatch 1.0 model card (Module 3)
├── reflection.md            # Profile comparison notes (Module 3)
└── requirements.txt         # anthropic, scikit-learn, numpy, pandas, pytest
```

---

## Requirements

```
anthropic>=0.97.0
scikit-learn
numpy
pandas
pytest
streamlit
```
