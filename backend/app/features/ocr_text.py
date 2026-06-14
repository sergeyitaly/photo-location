"""Optional OCR for signs and labels (free: Tesseract via pytesseract)."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def detect_text_in_image(image_array: np.ndarray, *, max_snippets: int = 8) -> Optional[List[str]]:
    """
    Extract short text snippets from an image. Returns None if OCR is unavailable or finds nothing.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    try:
        import pytesseract
    except ImportError:
        logger.debug("pytesseract not installed — OCR skipped (pip install pytesseract; system tesseract)")
        return None

    if image_array.ndim != 3 or image_array.shape[2] < 3:
        return None

    rgb = image_array[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    try:
        pil = Image.fromarray(rgb)
        raw = pytesseract.image_to_string(pil, lang="eng")
    except Exception as exc:
        logger.debug("Tesseract OCR failed: %s", exc)
        return None

    snippets: List[str] = []
    for line in (raw or "").splitlines():
        text = re.sub(r"\s+", " ", line).strip()
        if len(text) < 2:
            continue
        if len(text) > 120:
            text = text[:117] + "…"
        snippets.append(text)
        if len(snippets) >= max_snippets:
            break

    return snippets or None
