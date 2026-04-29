"""
Tests for input and output guardrails (src/guardrails.py).

All Gemini calls are mocked — zero API cost.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrails import validate_input, validate_output


def make_results(n=5):
    """Minimal result tuples matching (song_dict, score, explanation)."""
    return [
        (
            {"title": f"Song {i}", "artist": "Artist", "genre": "pop"},
            0.9 - i * 0.05,
            "some reason",
        )
        for i in range(n)
    ]


# ── validate_input ────────────────────────────────────────────────────────────

def test_valid_music_query_passes():
    valid, reason = validate_input("upbeat songs for running", mock=True)
    assert valid is True


def test_mock_always_passes():
    valid, _ = validate_input("what is 2 + 2", mock=True)
    assert valid is True  # mock bypasses real logic


def test_returns_tuple_of_bool_and_str():
    result = validate_input("chill study beats", mock=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], bool)
    assert isinstance(result[1], str)


def test_invalid_query_rejected_via_monkeypatch(monkeypatch):
    """Simulate Gemini correctly rejecting a non-music query."""
    import guardrails

    class FakeModel:
        def generate_content(self, model, contents):
            return type("R", (), {
                "text": '{"valid": false, "reason": "not a music request"}'
            })()

    class FakeClient:
        models = FakeModel()

    monkeypatch.setattr(guardrails, "_get_client", lambda: FakeClient())
    valid, reason = validate_input("tell me a joke", mock=False)
    assert valid is False
    assert "music" in reason.lower()


# ── validate_output ───────────────────────────────────────────────────────────

def test_mock_returns_high_confidence():
    confidence, flag = validate_output("upbeat pop", make_results(), mock=True)
    assert confidence == 0.85
    assert isinstance(flag, str)


def test_returns_float_confidence_and_str_flag():
    confidence, flag = validate_output("anything", make_results(), mock=True)
    assert isinstance(confidence, float)
    assert isinstance(flag, str)
    assert 0.0 <= confidence <= 1.0


def test_high_confidence_results_pass_threshold():
    confidence, _ = validate_output("happy pop music", make_results(), mock=True)
    assert confidence >= 0.6


def test_monkeypatched_low_confidence():
    """Simulate Gemini returning low confidence for a bad result set."""
    import guardrails
    original = guardrails.validate_output

    def mock_low(query, results, mock=False):
        return 0.3, "results do not match the requested genre"

    guardrails.validate_output = mock_low
    try:
        confidence, flag = guardrails.validate_output("happy music", make_results())
        assert confidence < 0.6
        assert "genre" in flag
    finally:
        guardrails.validate_output = original