# text_review_classifier.py
"""
Library-only text review classifier using Groq Llama 3.1 8B.

Usage:
    from text_review_classifier import classify_review, get_groq_client
    client = get_groq_client()  # optional reuse
    result = classify_review(
        category="Pizza restaurant",
        rating=1,
        text="Never been here but I heard it's terrible.",
        place_name="Slicetown",
        client=client
    )
    print(result)

Requires:
    pip install groq python-dotenv

Env:
    GROQ_API_KEY=gsk_********************************  (in .env or environment)
"""

from __future__ import annotations
import os, json, time
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from groq import Groq

# ----------------------------- Prompt -----------------------------

_SYSTEM_MSG = (
    "You are a strict review moderation classifier. "
    "Return STRICT JSON only (double-quoted keys/strings). "
    "Booleans must be lowercase true/false. No extra text."
)

_USER_TEMPLATE = """Task: Classify a location review for relevance and policy violations.

Definitions (pick exactly one violation):
- advertisement: contains promotional links/codes, phone numbers, coupons, “call now”, or unrelated marketing.
- irrelevant: off-topic; not about this place or its services; about another product/person/event; misplaced review (clearly for a different business type); filler/emoji-only.
- rant_no_visit: reviewer clearly states they did not visit (e.g., “never been”, “haven’t been”, “heard it’s…”).
- ok: none of the above AND the review is about this place.

Intent & fallback principle:
If none of the violation categories apply and the review is about the place, classify as "ok".
Default to "ok" when in doubt, because the goal is to increase user trust, ensure fair business representation, and enhance platform credibility.

Output STRICT JSON only (double-quoted keys/strings; booleans lowercase):
{{
  "relevant": true|false,
  "violation": "advertisement"|"irrelevant"|"rant_no_visit"|"ok",
  "classification": "advertisement"|"irrelevant"|"rant_no_visit"|"ok",
  "confidence": 0.0-1.0,
  "reasoning": "<short phrase>",
  "indicators": ["token1","token2"]
}}

Data:
Place Name: {place_name}
Place Type: {place_type}
Rating: {rating}
Text: "{review_text}"
"""

# ----------------------------- Public API -----------------------------

def get_groq_client() -> Groq:
    """Create a Groq client using GROQ_API_KEY from .env / environment."""
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Missing GROQ_API_KEY. Put it in .env or as an environment variable.")
    return Groq(api_key=key)


def classify_review(
    category: str,
    rating: Optional[float],
    text: str,
    place_name: str = "",
    *,
    client: Optional[Groq] = None,
    max_retries: int = 2,
    retry_sleep: float = 0.8,
) -> Dict[str, Any]:
    """
    Classify a single text-only review.

    Args:
        category: Business category (place type), e.g. "Pizza restaurant".
        rating: Numeric rating if available (can be None).
        text: Review text.
        place_name: Optional business name for extra grounding.
        client: Optional Groq client instance; if None, one will be created.
        max_retries: How many times to retry on transient API errors.
        retry_sleep: Base sleep (seconds) for exponential backoff.

    Returns:
        dict with keys:
        {
          "relevant": true/false | None on error,
          "violation": "advertisement"|"irrelevant"|"rant_no_visit"|"ok"|"LLM_ERROR",
          "classification": same as violation or "LLM_ERROR",
          "confidence": float | None,
          "reasoning": str,
          "indicators": list[str]
        }
    """
    if client is None:
        client = get_groq_client()

    # Basic sanitation: cap very long inputs to keep prompt small and stable
    safe_text = (text or "").strip()
    if len(safe_text) > 2000:
        safe_text = safe_text[:2000] + "…"

    user_msg = _USER_TEMPLATE.format(
        place_name=place_name or "",
        place_type=category or "",
        rating=rating if rating is not None else "",
        review_text=safe_text.replace('"', '\\"'),
    )

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _SYSTEM_MSG},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,  # deterministic for labeling
                response_format={"type": "json_object"},  # strict JSON
                max_tokens=300,
            )
            return json.loads(resp.choices[0].message.content)

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(retry_sleep * (2 ** attempt))
            else:
                # Always return a consistent shape on failure
                return {
                    "relevant": None,
                    "violation": "LLM_ERROR",
                    "classification": "LLM_ERROR",
                    "confidence": None,
                    "reasoning": f"Exception: {type(last_err).__name__}: {last_err}",
                    "indicators": [],
                }

__all__ = ["classify_review", "get_groq_client"]