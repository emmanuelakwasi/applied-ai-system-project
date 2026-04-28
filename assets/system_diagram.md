# VibeMatch System Diagram

```mermaid
flowchart TD

    %% ── INPUT ──────────────────────────────────────────────────
    USER([" Human User "])
    USER -->|"natural language query\ne.g. 'chill music for studying'\nor --profile / --classic flag"| MAIN

    %% ── ENTRY POINT ────────────────────────────────────────────
    MAIN["main.py\nEntry Point\n(argparse)"]

    %% ── DATA SOURCE ────────────────────────────────────────────
    CSV[("songs.csv\n10 songs · 9 attributes\ngenre, mood, energy,\ntempo, valence,\ndanceability, acousticness")]

    %% ── STEP 1 : RULE-BASED SCORING ────────────────────────────
    MAIN --> INFER["Profile Inferrer\nkeyword → UserProfile\ngenre / mood / energy"]
    CSV  --> SCORER
    INFER -->|"UserProfile"| SCORER["Rule-Based Scorer\nrecommender.py\n+1.0  genre match\n+1.0  mood match\n+0–4  energy similarity"]
    SCORER -->|"ranked songs\n+ score breakdowns"| RAG

    %% ── STEP 2 : RAG SYNTHESIS ─────────────────────────────────
    CSV --> RETRIEVER["TF-IDF Retriever\nSongKnowledgeBase\nbigram cosine similarity"]
    RETRIEVER -->|"top-k semantic matches\n(independent of profile)"| RAG

    RAG["RAGAssistant\nrecommend_with_scores()\ncross-references rule scores\n+ semantic ranks"]

    RAG -->|"structured prompt:\nscored list · semantic list\nscore breakdowns · all attributes"| CACHE
    CACHE["Prompt Cache\nsystem prompt cached\n(role + full catalog)\nreused across queries"]
    CACHE --> CLAUDE["Claude  claude-opus-4-7\nadaptive thinking · streaming\ncites scores, flags gaps,\nflags attribute blind spots"]

    CLAUDE -->|"streamed recommendation"| USER

    %% ── CLASSIC PATH (no AI) ────────────────────────────────────
    MAIN -->|"--classic flag"| SCORES["Scored List\nprinted to terminal\n(no API key needed)"]
    SCORES -->|"human compares\nscores vs AI output"| USER

    %% ── TESTING LAYER ───────────────────────────────────────────
    T1[/"test_recommender.py\n· sort order correct\n· score values accurate\n· explanation non-empty"/]
    T2[/"test_rag_pipeline.py\n· returns k results\n· genre/mood queries match\n· dict keys present"/]

    T1 -. "pytest" .-> SCORER
    T2 -. "pytest" .-> RETRIEVER

    %% ── STYLES ──────────────────────────────────────────────────
    classDef human   fill:#fef9c3,stroke:#ca8a04,color:#000
    classDef data    fill:#dbeafe,stroke:#2563eb,color:#000
    classDef ai      fill:#dcfce7,stroke:#16a34a,color:#000
    classDef cache   fill:#f0fdf4,stroke:#86efac,color:#000
    classDef test    fill:#fef2f2,stroke:#dc2626,color:#000
    classDef core    fill:#fff7ed,stroke:#ea580c,color:#000
    classDef output  fill:#f5f3ff,stroke:#7c3aed,color:#000

    class USER human
    class CSV data
    class CLAUDE,RAG ai
    class CACHE cache
    class T1,T2 test
    class SCORER,RETRIEVER,INFER,MAIN core
    class SCORES output
```

## Component Reference

| Component | File | Role |
|---|---|---|
| **Entry Point** | `src/main.py` | Parses args, orchestrates both steps, handles fallback |
| **Profile Inferrer** | `src/main.py` | Maps query keywords to a discrete UserProfile |
| **Rule-Based Scorer** | `src/recommender.py` | Scores every song: genre (+1), mood (+1), energy (0–4) |
| **TF-IDF Retriever** | `src/rag_pipeline.py` · `SongKnowledgeBase` | Bigram cosine similarity over song metadata |
| **RAG Assistant** | `src/rag_pipeline.py` · `RAGAssistant` | Combines both signals into a structured Claude prompt |
| **Prompt Cache** | Anthropic API | Caches stable system prompt (role + catalog) across queries |
| **Claude** | `claude-opus-4-7` | Reasons over scores + retrieval, streams recommendation |
| **Knowledge Base** | `data/songs.csv` | 10 songs × 9 attributes; source for both scorer and retriever |

## Data Flow Summary

```
Query
  → Profile Inferrer          (keyword heuristic → genre/mood/energy)
      → Rule-Based Scorer     (scores all 10 songs)
  → TF-IDF Retriever          (cosine similarity → top-3 semantic matches)
      → RAGAssistant           (merges both; builds structured prompt)
          → Claude API         (reasons over evidence; streams output)
              → Human          (reads streamed recommendation)
```

## Testing & Human Verification

- **`test_recommender.py`** — automated: checks scorer sort order and score correctness
- **`test_rag_pipeline.py`** — automated: checks retriever returns right genres/counts/keys
- **`--classic` flag** — human verification: lets a person compare bare scores against AI output side-by-side to spot hallucinations or score/recommendation mismatches
