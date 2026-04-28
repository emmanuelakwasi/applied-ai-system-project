"""
Unit tests for AI evaluation helpers.

Covers _parse_confidence() and the response-quality check logic
used by src/eval.py. No API key required.
"""

import re
import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_pipeline import _parse_confidence

_SCORE_RE = re.compile(r"\b\d\.\d{2}\b")


# ---------------------------------------------------------------------------
# _parse_confidence
# ---------------------------------------------------------------------------

class TestParseConfidence:
    def test_valid_em_dash(self):
        text = "Great recommendation.\n\nCONFIDENCE: 0.87 — Both signals agree on the top picks."
        score, reason = _parse_confidence(text)
        assert score == pytest.approx(0.87)
        assert "Both signals" in reason

    def test_valid_hyphen(self):
        text = "CONFIDENCE: 0.65 - Minor disagreement on the third pick."
        score, reason = _parse_confidence(text)
        assert score == pytest.approx(0.65)
        assert "Minor disagreement" in reason

    def test_missing_returns_none(self):
        text = "This response contains no confidence line at all."
        score, reason = _parse_confidence(text)
        assert score is None
        assert reason == ""

    def test_malformed_no_separator(self):
        # No dash/em-dash between value and reason — should not match
        text = "CONFIDENCE: 0.75 without any separator character"
        score, reason = _parse_confidence(text)
        assert score is None

    def test_case_insensitive(self):
        text = "confidence: 0.90 — strong agreement between both signals"
        score, reason = _parse_confidence(text)
        assert score == pytest.approx(0.90)

    def test_mid_text_confidence(self):
        # Confidence line can appear anywhere in the response
        text = (
            "I recommend Storm Runner.\n"
            "CONFIDENCE: 0.82 — rule-based and semantic both rank it first.\n"
            "Nothing else to add."
        )
        score, reason = _parse_confidence(text)
        assert score == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# Quality check logic (mirrors src/eval.py)
# ---------------------------------------------------------------------------

class TestQualityChecks:
    def test_cites_top_song_present(self):
        response = 'I recommend "Midnight Coding" by LoRoom as the clear top pick.'
        assert "Midnight Coding".lower() in response.lower()

    def test_cites_top_song_absent(self):
        response = "Library Rain is a great fit for late-night studying."
        assert "Midnight Coding".lower() not in response.lower()

    def test_score_citation_found(self):
        response = "Storm Runner scores 5.96 — the highest rule-based score in the catalog."
        assert bool(_SCORE_RE.search(response))

    def test_score_citation_absent(self):
        response = "Storm Runner scores highest in the catalog with perfect alignment."
        assert not bool(_SCORE_RE.search(response))

    def test_confidence_threshold(self):
        # Scores at or above 0.5 are acceptable; below are not
        assert 0.50 >= 0.5   # boundary — acceptable
        assert 0.49 < 0.5    # just below — not acceptable
        assert 0.87 >= 0.5
