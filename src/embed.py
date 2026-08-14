"""
Embedding functions for TailorTalk saree similarity search.

Two signals, fused:
1. DINOv2 embedding — captures fine-grained visual texture/pattern (better than
   CLIP for this since CLIP is semantic/text-aligned, not texture-sensitive).
2. HSV color histogram — captures color-family closeness explicitly, since two
   sarees can be DINOv2-similar in weave/pattern but different color families.

Background removal (rembg) is applied first since the dataset mixes model
shots, flat-lay shots, and close-ups — without this, background/mannequin/
lighting noise dominates the embedding and different sarees score too close
to similar ones.
"""

import numpy as np
import cv2
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from rembg import remove, new_session

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MODEL_NAME = "facebook/dinov2-base"
_processor = None
_model = None
_rembg_session = None


def _load_model():
    """Lazy-load DINOv2 so importing this module is cheap."""
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(_MODEL_NAME)
        _model = AutoModel.from_pretrained(_MODEL_NAME).to(DEVICE)
        _model.eval()
    return _processor, _model


def _load_rembg():
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("u2net")
    return _rembg_session


def remove_background(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """
    Removes background, composites the foreground onto a plain white canvas
    (DINOv2 was trained on natural images, transparent/black backgrounds can
    introduce their own artifacts). Returns (rgb_image, alpha_mask) — the mask
    is reused so the color histogram only looks at foreground pixels too.
    """
    session = _load_rembg()
    rgba = remove(image.convert("RGB"), session=session)  # RGBA output
    rgba_np = np.array(rgba)
    alpha = rgba_np[:, :, 3]

    white_bg = np.ones_like(rgba_np[:, :, :3]) * 255
    alpha_norm = (alpha / 255.0)[:, :, None]
    composited = (rgba_np[:, :, :3] * alpha_norm + white_bg * (1 - alpha_norm)).astype(np.uint8)

    return Image.fromarray(composited), alpha


def get_dinov2_embedding(image: Image.Image) -> np.ndarray:
    """Returns a normalized DINOv2 CLS embedding for a PIL image."""
    processor, model = _load_model()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token = outputs.last_hidden_state[:, 0, :]
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    norm = np.linalg.norm(cls_embedding)
    return cls_embedding / norm if norm > 0 else cls_embedding


def get_color_histogram(image: Image.Image, alpha_mask: np.ndarray = None, bins=(8, 12, 3)) -> np.ndarray:
    """
    HSV histogram, weighted toward Hue (color family) more than Value (lighting).
    If alpha_mask is given, only foreground pixels contribute — otherwise
    background (white canvas, mannequin skin, etc.) pollutes the color signal.
    bins = (H_bins, S_bins, V_bins).
    """
    img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = None
    if alpha_mask is not None:
        # threshold: only count pixels that were actually foreground
        mask = (alpha_mask > 30).astype(np.uint8)

    hist = cv2.calcHist(
        [img_hsv], [0, 1, 2], mask, list(bins),
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist


def get_fused_embedding(image: Image.Image) -> dict:
    """
    Removes background once, then derives both embeddings from the same
    foreground-isolated image so DINOv2 and color histogram agree on what
    counts as "the saree".
    """
    fg_image, alpha_mask = remove_background(image)
    return {
        "dino": get_dinov2_embedding(fg_image),
        "color": get_color_histogram(fg_image, alpha_mask=alpha_mask),
    }


def fused_similarity(query: dict, candidate: dict, alpha: float = 0.7, beta: float = 0.3) -> float:
    """
    Combines DINOv2 cosine similarity with color histogram intersection.
    alpha/beta are the fusion weights — tune these after eyeballing results
    on your actual dataset (see test script).
    """
    dino_sim = float(np.dot(query["dino"], candidate["dino"]))  # both pre-normalized
    color_sim = float(cv2.compareHist(
        query["color"].astype(np.float32),
        candidate["color"].astype(np.float32),
        cv2.HISTCMP_INTERSECT,
    ))
    return alpha * dino_sim + beta * color_sim


if __name__ == "__main__":
    # Quick standalone sanity check — point at any two local images to test.
    import sys
    if len(sys.argv) == 3:
        img1 = Image.open(sys.argv[1])
        img2 = Image.open(sys.argv[2])
        e1 = get_fused_embedding(img1)
        e2 = get_fused_embedding(img2)
        print("DINOv2 cosine sim:", np.dot(e1["dino"], e2["dino"]))
        print("Color hist sim:", cv2.compareHist(
            e1["color"].astype(np.float32), e2["color"].astype(np.float32), cv2.HISTCMP_INTERSECT
        ))
        print("Fused score:", fused_similarity(e1, e2))
    else:
        print("Usage: python embed.py <image1> <image2>")