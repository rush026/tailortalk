"""
LangChain-compatible tool for saree visual similarity search.

Loads the pre-built FAISS index + color histograms + product catalog,
then exposes a single function `find_similar_sarees` that:
  1. Takes an image (file path, URL, or raw bytes)
  2. Removes background, extracts DINOv2 + HSV embeddings
  3. Queries FAISS for DINOv2-nearest candidates
  4. Re-ranks using fused score (DINOv2 cosine + color histogram intersection)
  5. Returns enriched product results with scores
"""

import json
import io
import os
import base64
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
import cv2
import requests
from PIL import Image
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Paths — resolve relative to project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = _PROJECT_ROOT / "data" / "index"
IMAGES_DIR = _PROJECT_ROOT / "data" / "images"

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------
_faiss_index = None
_color_hists = None
_metadata = None
_product_catalog = None


def _load_index():
    global _faiss_index, _color_hists, _metadata, _product_catalog

    if _faiss_index is None:
        _faiss_index = faiss.read_index(str(INDEX_DIR / "dino.index"))
        _color_hists = np.load(str(INDEX_DIR / "color_hists.npy"))
        with open(INDEX_DIR / "metadata.json") as f:
            _metadata = json.load(f)
        catalog_path = INDEX_DIR / "product_catalog.json"
        if catalog_path.exists():
            with open(catalog_path) as f:
                _product_catalog = json.load(f)
        else:
            _product_catalog = {}

    return _faiss_index, _color_hists, _metadata, _product_catalog


def _load_image(image_source: str) -> Image.Image:
    """Load image from a file path, URL, or base64 string."""
    # Try as local file path first
    path = Path(image_source)
    if path.exists():
        return Image.open(path).convert("RGB")

    # Also try relative to project root
    project_path = _PROJECT_ROOT / image_source
    if project_path.exists():
        return Image.open(project_path).convert("RGB")

    # Try as URL
    if image_source.startswith(("http://", "https://")):
        resp = requests.get(image_source, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    # Try as base64
    try:
        img_data = base64.b64decode(image_source)
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception:
        pass

    raise ValueError(f"Cannot load image from: {image_source[:100]}...")


def _sku_from_filename(filename: str) -> str:
    """Extract SKU from filename like 'QS204820.jpg' or 'Cotton Saree Brown Bird Design AA207916.jpg'."""
    stem = Path(filename).stem
    # If the stem itself is a valid SKU (no spaces), use it directly
    if " " not in stem:
        return stem
    # Otherwise take the last token (the SKU is usually appended at the end)
    parts = stem.split()
    return parts[-1] if parts else stem


def search_similar(image_source: str, top_k: int = 5) -> list[dict]:
    """
    Core search function. Returns a list of dicts with match details.
    """
    from src.embed import get_fused_embedding

    index, color_hists, metadata, catalog = _load_index()

    # 1. Load and embed the query image
    query_image = _load_image(image_source)
    query_emb = get_fused_embedding(query_image)
    query_dino = query_emb["dino"].reshape(1, -1).astype("float32")
    query_color = query_emb["color"].astype("float32")

    # 2. FAISS search — retrieve extra candidates for re-ranking
    retrieve_k = min(top_k * 4, index.ntotal)
    dino_scores, dino_indices = index.search(query_dino, retrieve_k)

    # 3. Fused re-ranking
    alpha, beta = 0.7, 0.3
    candidates = []

    for rank in range(retrieve_k):
        idx = int(dino_indices[0][rank])
        if idx < 0:
            continue

        dino_sim = float(dino_scores[0][rank])  # already cosine (IP on normalized)

        # Color histogram intersection
        cand_color = color_hists[idx].astype("float32")
        color_sim = float(cv2.compareHist(
            query_color, cand_color, cv2.HISTCMP_INTERSECT
        ))

        fused = alpha * dino_sim + beta * color_sim

        candidates.append({
            "faiss_idx": idx,
            "dino_score": round(dino_sim, 4),
            "color_score": round(color_sim, 4),
            "fused_score": round(fused, 4),
        })

    # Sort by fused score descending
    candidates.sort(key=lambda x: x["fused_score"], reverse=True)
    top_candidates = candidates[:top_k]

    # 4. Enrich with metadata + product catalog
    results = []
    for c in top_candidates:
        idx_str = str(c["faiss_idx"])
        meta = metadata.get(idx_str, {})
        filename = meta.get("filename", "unknown.jpg")
        sku = _sku_from_filename(filename)
        product = catalog.get(sku, {})

        image_path = meta.get("path", "")

        results.append({
            "rank": len(results) + 1,
            "sku": sku,
            "name": product.get("name", f"Saree {sku}"),
            "similarity_score": c["fused_score"],
            "dino_score": c["dino_score"],
            "color_score": c["color_score"],
            "retail_price": product.get("retail_price", "N/A"),
            "discounted_price": product.get("discounted_price", "N/A"),
            "in_stock": product.get("in_stock", False),
            "website_link": product.get("website_link", ""),
            "image_url": product.get("image_url", ""),
            "local_image_path": str(_PROJECT_ROOT / image_path) if image_path else "",
            "filename": filename,
        })

    return results


@tool
def find_similar_sarees(
    image_source: str,
    top_k: int = 5,
) -> str:
    """Find visually similar sarees from the catalog given a query image.

    Use this tool whenever the user uploads an image, provides an image URL,
    or asks to find sarees similar to a given image.

    Args:
        image_source: Path to a local image file, or a URL to an image.
        top_k: Number of similar sarees to return (default 5, max 10).

    Returns:
        A formatted string describing the top matching sarees with their
        similarity scores, names, prices, and links.
    """
    top_k = min(max(top_k, 1), 10)

    try:
        results = search_similar(image_source, top_k=top_k)
    except Exception as e:
        return f"Error during search: {str(e)}"

    if not results:
        return "No similar sarees found. The image may not be processable."

    # Format results for the LLM to present naturally
    lines = [f"Found {len(results)} similar sarees:\n"]

    for r in results:
        score_pct = round(r["similarity_score"] * 100, 1)
        stock_str = "In Stock" if r["in_stock"] else "Out of Stock"
        price_str = f"₹{r['discounted_price']}"
        if r["retail_price"] != r["discounted_price"] and r["retail_price"] != "N/A":
            price_str = f"₹{r['discounted_price']} (MRP ₹{r['retail_price']})"

        lines.append(
            f"#{r['rank']}. {r['name']}\n"
            f"   SKU: {r['sku']} | Match: {score_pct}% | Price: {price_str} | {stock_str}\n"
            f"   Link: {r['website_link']}\n"
        )

    return "\n".join(lines)
