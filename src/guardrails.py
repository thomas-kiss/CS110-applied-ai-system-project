"""
Input and output guardrails using Gemini.

Both functions accept mock=True for test environments — no API calls are made.
Real calls use gemma-3-27b-it and expect GEMINI_API_KEY in the environment.
"""

import json
import logging
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "gemma-3-27b-it"
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from a Gemini response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def validate_input(query: str, mock: bool = False) -> tuple[bool, str]:
    """
    Reject queries that are not music recommendation requests.
    Returns (is_valid, reason).
    """
    if mock:
        return True, "valid"

    prompt = (
        'Is the following a music recommendation request?\n'
        f'Query: "{query}"\n\n'
        'Reply with JSON only — no explanation outside the JSON.\n'
        'Format: {"valid": true/false, "reason": "one sentence"}'
    )
    try:
        response = _get_client().models.generate_content(
            model=MODEL, contents=prompt,
        )
        data = _parse_json(response.text)
        return bool(data["valid"]), str(data.get("reason", ""))
    except Exception as exc:
        logger.warning(
            "validate_input parse error: %s — defaulting to valid", exc
        )
        return True, "parse error, defaulting to valid"


def validate_output(
    query: str,
    results: list,
    mock: bool = False,
) -> tuple[float, str]:
    """
    Score how well the top results match the original query (0.0 – 1.0).
    Returns (confidence, flag_message).
    """
    if mock:
        return 0.85, "good match"

    summary = "\n".join(
        f"  - {r[0]['title']} by {r[0]['artist']} "
        f"(genre: {r[0]['genre']}, "
        f"score: {r[1]:.2f})"
        for r in results
    )
    prompt = (
        f'User request: "{query}"\n\n'
        f"Top recommendations:\n{summary}\n\n"
        "Rate how well these recommendations match the request "
        "(0.0 = terrible, 1.0 = perfect).\n"
        "Reply with JSON only.\n"
        'Format: {"confidence": 0.0-1.0, '
        '"flag": "one sentence explaining the score"}'
    )
    try:
        response = _get_client().models.generate_content(
            model=MODEL, contents=prompt,
        )
        data = _parse_json(response.text)
        return float(data["confidence"]), str(data.get("flag", ""))
    except Exception as exc:
        logger.warning(
            "validate_output parse error: %s — defaulting to 0.5", exc
        )
        return 0.5, "parse error, confidence unknown"
