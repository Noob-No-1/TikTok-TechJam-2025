# image_augmented_classifier.py
from __future__ import annotations
import os
import json
from typing import Optional, Dict, Any

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# 复用你之前的文本分类函数
from review_classifier import classify_review, get_groq_client

# --------- 1) 简单的图片描述（BLIP） ----------
# 第一次调用会从 HF 下载模型；可改成 large 版提升质量:
# "Salesforce/blip-image-captioning-large"
_BLIP_MODEL_ID = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-base")
_blip_proc: Optional[BlipProcessor] = None
_blip: Optional[BlipForConditionalGeneration] = None

def _ensure_blip():
    global _blip_proc, _blip
    if _blip is None:
        # Use safetensors-only to avoid CVE-2025-32434 requirement for torch>=2.6 when loading .bin
        _blip_proc = BlipProcessor.from_pretrained(_BLIP_MODEL_ID)
        _blip = BlipForConditionalGeneration.from_pretrained(
            _BLIP_MODEL_ID,
            use_safetensors=True,   # force .safetensors weights
            trust_remote_code=True  # allow model-defined code if needed
        )

def caption_image(image_path: str, max_new_tokens: int = 40) -> str:
    _ensure_blip()
    img = Image.open(image_path).convert("RGB")
    inputs = _blip_proc(img, return_tensors="pt")
    out = _blip.generate(**inputs, max_new_tokens=max_new_tokens)
    cap = _blip_proc.decode(out[0], skip_special_tokens=True)
    # 简单清洗 & 截断，避免 prompt 太长
    cap = cap.strip()
    if len(cap) > 300:
        cap = cap[:300] + "…"
    return cap

# --------- 2) 对外接口：带图的分类 ----------

_IMG_SYSTEM_MSG = (
    "You are a strict review moderation classifier for multimodal reviews (text + image caption). "
    "Return STRICT JSON only (double-quoted keys/strings). Booleans must be lowercase true/false. No extra text."
)

_IMG_USER_TEMPLATE = """Task: Classify a location review with an attached image for relevance and policy violations.

Definitions (pick exactly one violation):
- advertisement: text contains promo/links/coupons/phone numbers OR the image content is clearly marketing.
- irrelevant: text is off-topic (about another product/person/event), misplaced review, filler/emoji-only.
- rant_no_visit: reviewer clearly states they did not visit (e.g., “never been”, “haven’t been”, “heard it’s…”).
- image_irrelevant: the image content (per caption) is unrelated to the business category or the review content.
- image_advertisement: the image appears to be an ad or promo (e.g., poster/flyer/QR/coupon/brand banner/phone number).
- ok: none of the above AND the review is about this place.

Intent & fallback principle:
If none of the categories apply and the review is about the place, classify as "ok".

Output STRICT JSON only:
{{
  "relevant": true|false,
  "violation": "advertisement"|"irrelevant"|"rant_no_visit"|"image_irrelevant"|"image_advertisement"|"ok",
  "classification": "advertisement"|"irrelevant"|"rant_no_visit"|"image_irrelevant"|"image_advertisement"|"ok",
  "confidence": 0.0-1.0,
  "reasoning": "<short phrase>",
  "indicators": ["token1","token2"]
}}

Data:
Place Name: {place_name}
Place Type (category): {place_type}
Rating: {rating}
Review Text: "{review_text}"
Image Caption: "{image_caption}"
Heuristic Hints: {heuristics}
"""

def classify_review_with_image(
    category: str,
    rating: Optional[float],
    text: str,
    image_path: str,
    place_name: str = "",
    *,
    client=None,
    caption_prefix: str = "Photo shows: ",
) -> Dict[str, Any]:
    """
    Generate an image caption -> append to the review text in a plain sentence, then reuse the text-only classifier.
    Example augmentation: "... With a picture of <caption>."
    """
    cap = caption_image(image_path)
    if client is None:
        client = get_groq_client()
    base_text = (text or "").strip()
    augmented_text = (base_text + (" " if base_text else "") + f"With a picture of {cap}.").strip()
    out = classify_review(
        category=category,
        rating=rating,
        text=augmented_text,
        place_name=place_name,
        client=client,
    )
    # attach debug info
    try:
        out["_image_caption"] = cap
    except Exception:
        pass
    return out
# review_classifier.py
# (only the relevant part of the file is shown with the modified _USER_TEMPLATE)

_USER_TEMPLATE = """Task: Classify a location review for relevance and policy violations.

Definitions (pick exactly one violation):
- advertisement: text contains promo/links/coupons/phone numbers.
- irrelevant: text is off-topic (about another product/person/event), misplaced review, filler/emoji-only.
- rant_no_visit: reviewer clearly states they did not visit (e.g., “never been”, “haven’t been”, “heard it’s…”).
- ok: none of the above AND the review is about this place.

Image hint (optional): If the review text ends with a sentence like "With a picture of <caption>", treat that as the image description. If the image description contradicts or is unrelated to the place/category or the review content, classify as "irrelevant" and add an indicator token "image_mismatch". If the image description looks promotional (poster/QR/coupon/banner/phone number/discount/promo), classify as "advertisement" and add an indicator token "image_ad". Include a brief reasoning mentioning the mismatch or ad-like content.

Output STRICT JSON only:
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
Place Type (category): {place_type}
Rating: {rating}
Review Text: "{review_text}"
"""