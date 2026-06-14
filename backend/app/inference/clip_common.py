"""Shared CLIP load + image–text softmax for cue panels (no coordinate regression)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_clip_runtime_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=8)
def load_clip(model_id: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, processor, device


def clip_softmax_for_prompts(
    image_rgb: np.ndarray,
    prompts: List[str],
    *,
    model_id: str,
) -> List[Tuple[str, float]]:
    """
    Returns (prompt, probability) pairs sorted by probability descending.
    Probabilities softmax over `prompts` only for this image.
    """
    if not prompts:
        return []
    if not is_clip_runtime_available():
        return [(p, 1.0 / len(prompts)) for p in prompts]

    import torch
    import torch.nn.functional as F

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    model, processor, device = load_clip(model_id)
    inputs = processor(text=prompts, images=pil, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits_per_image[0].float()
        probs = F.softmax(logits, dim=0).cpu().numpy()

    pairs = [(prompts[i], float(probs[i])) for i in range(len(prompts))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def clip_softmax_probs_ordered(
    image_rgb: np.ndarray,
    text_prompts: List[str],
    model_id: str,
    *,
    max_texts_per_forward: int = 56,
) -> Optional[np.ndarray]:
    """
    Softmax probabilities over `text_prompts` in input order (for zero-shot indexing).
    Long prompt lists are split across forwards; logits are concatenated then softmax once
    (same image embedding used per chunk — comparable scores).
    Returns None if CLIP cannot run.
    """
    if not text_prompts:
        return None
    if not is_clip_runtime_available():
        return None

    import torch
    import torch.nn.functional as F
    from PIL import Image

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    model, processor, device = load_clip(model_id)
    chunk = max(8, min(max_texts_per_forward, 256))

    if len(text_prompts) <= chunk:
        inputs = processor(text=text_prompts, images=pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0].float()
            probs = F.softmax(logits, dim=0).cpu().numpy()
        return probs

    parts: List[torch.Tensor] = []
    for start in range(0, len(text_prompts), chunk):
        batch = text_prompts[start : start + chunk]
        inputs = processor(text=batch, images=pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits_per_image[0].float()
        parts.append(logits.cpu())
    logits_full = torch.cat(parts)
    probs = F.softmax(logits_full, dim=0).numpy()
    return probs


def clip_image_text_cosine_similarity(
    image_rgb: np.ndarray, text: str, model_id: str
) -> Optional[float]:
    """
    Normalized CLIP cosine similarity between image and a text passage (e.g. Wikipedia lead).
    Returns a score in ~[0, 1] via ((cos+1)/2); None if CLIP unavailable or empty text.
    """
    if not text or not str(text).strip():
        return None
    if not is_clip_runtime_available():
        return None

    import torch
    import torch.nn.functional as F

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    model, processor, device = load_clip(model_id)
    txt = str(text).strip()
    if len(txt) > 8000:
        txt = txt[:8000]

    img_inputs = processor(images=pil, return_tensors="pt")
    txt_inputs = processor(text=[txt], return_tensors="pt", padding=True, truncation=True)
    img_inputs = {k: v.to(device) for k, v in img_inputs.items()}
    txt_inputs = {k: v.to(device) for k, v in txt_inputs.items()}
    with torch.no_grad():
        img_f = model.get_image_features(**img_inputs)
        txt_f = model.get_text_features(**txt_inputs)
        img_f = F.normalize(img_f, dim=-1)
        txt_f = F.normalize(txt_f, dim=-1)
        cos = (img_f @ txt_f.T).squeeze().item()
    return float(max(0.0, min(1.0, (float(cos) + 1.0) / 2.0)))


def encode_image_embedding(image_rgb: np.ndarray, model_id: str) -> Optional[np.ndarray]:
    """L2-normalized CLIP image embedding, or None if CLIP unavailable."""
    if not is_clip_runtime_available():
        return None
    import torch

    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    model, processor, device = load_clip(model_id)
    inputs = processor(images=pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()


def clip_image_image_cosine_similarity(
    image_a: np.ndarray,
    image_b: np.ndarray,
    model_id: str,
) -> Optional[float]:
    """
    Cosine similarity between two images in CLIP embedding space (~[-1, 1]).
    Used for Wikipedia Commons / lead-image photo matching.
    """
    emb_a = encode_image_embedding(image_a, model_id)
    emb_b = encode_image_embedding(image_b, model_id)
    if emb_a is None or emb_b is None:
        return None
    return float(np.dot(emb_a, emb_b))
