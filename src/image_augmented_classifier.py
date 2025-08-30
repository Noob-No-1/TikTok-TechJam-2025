# image_augmented_classifier.py
"""
Image-augmented review classification helper.

Responsibilities:
- Generate an image caption with BLIP.
- Append the caption to the provided review text as:
  "... With a picture of <caption>."
- Delegate the final decision to `review_classifier.classify_review`.
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from review_classifier import classify_review, get_groq_client

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLIP_MODEL_ID = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
CAPTION_MAX_TOKENS = int(os.getenv("CAPTION_MAX_TOKENS", "40"))
CAPTION_MAX_CHARS = int(os.getenv("CAPTION_MAX_CHARS", "300"))

# Lazy globals used by the captioner
_blip_proc: Optional[BlipProcessor] = None
_blip: Optional[BlipForConditionalGeneration] = None


def _ensure_blip() -> None:
    """Lazily load the BLIP captioning model and processor."""
    global _blip_proc, _blip
    if _blip is not None and _blip_proc is not None:
        return
    _blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
    _blip = BlipForConditionalGeneration.from_pretrained(
        BLIP_MODEL_ID,
        use_safetensors=True,  # prefer .safetensors weights
        trust_remote_code=True,
    )


def caption_image(image_path: str, max_new_tokens: Optional[int] = None) -> str:
    """
    Generate a short caption for an image using BLIP.

    Args:
        image_path: Path to a local image file.
        max_new_tokens: Optional override for generation length.

    Returns:
        A cleaned caption string.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        ValueError: If the image cannot be opened.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    _ensure_blip()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Unable to open image '{image_path}': {exc}") from exc

    tokens = max_new_tokens or CAPTION_MAX_TOKENS
    inputs = _blip_proc(img, return_tensors="pt")
    output = _blip.generate(**inputs, max_new_tokens=tokens)
    caption = _blip_proc.decode(output[0], skip_special_tokens=True).strip()

    if len(caption) > CAPTION_MAX_CHARS:
        caption = caption[:CAPTION_MAX_CHARS] + "…"

    return caption


def classify_review_with_image(
    category: str,
    rating: Optional[float],
    text: str,
    image_path: str,
    place_name: str = "",
    *,
    client=None,
) -> Dict[str, Any]:
    """
    Classify a review that includes an image.

    Steps:
    1) Caption the image with BLIP.
    2) Append the caption to the text: "... With a picture of <caption>.".
    3) Call the text-only ``classify_review`` to get the final decision.

    Args:
        category: Business category (e.g., "Pizza restaurant").
        rating: Star rating if available.
        text: The review text.
        image_path: Local path to the attached image.
        place_name: Optional business name.
        client: Optional Groq client; if not provided, one is created.

    Returns:
        The JSON-like dict from ``classify_review`` with an added ``_image_caption`` field.
    """
    caption = caption_image(image_path)

    base_text = (text or "").strip()
    augmented_text = (base_text + (" " if base_text else "") + f"With a picture of {caption}.").strip()

    if client is None:
        client = get_groq_client()

    result = classify_review(
        category=category,
        rating=rating,
        text=augmented_text,
        place_name=place_name,
        client=client,
    )

    # Attach debug info for audits (no-op if result is not a dict)
    if isinstance(result, dict):
        result["_image_caption"] = caption

    return result


__all__ = [
    "caption_image",
    "classify_review_with_image",
]