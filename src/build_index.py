"""
Builds the FAISS index + metadata store from all images in data/images/.

Run once (or whenever the dataset changes):
    python src/build_index.py

Produces:
    data/index/dino.index      -- FAISS index over DINOv2 embeddings
    data/index/color_hists.npy -- color histograms, aligned by row to dino.index
    data/index/metadata.json   -- id -> {filename, path}
"""

import json
from pathlib import Path
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm

from embed import get_fused_embedding

IMAGES_DIR = Path("data/images")
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build():
    image_paths = sorted(
        p for p in IMAGES_DIR.glob("*.jpg")
    )
    print(f"Found {len(image_paths)} images to index.")

    dino_vecs = []
    color_vecs = []
    metadata = {}

    for i, path in enumerate(tqdm(image_paths, desc="Embedding")):
        try:
            img = Image.open(path)
            emb = get_fused_embedding(img)
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue

        dino_vecs.append(emb["dino"])
        color_vecs.append(emb["color"])
        metadata[str(i)] = {"filename": path.name, "path": str(path)}

    dino_matrix = np.stack(dino_vecs).astype("float32")
    color_matrix = np.stack(color_vecs).astype("float32")

    # Inner product on pre-normalized DINOv2 vectors == cosine similarity
    dim = dino_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dino_matrix)

    faiss.write_index(index, str(INDEX_DIR / "dino.index"))
    np.save(INDEX_DIR / "color_hists.npy", color_matrix)
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nIndexed {len(metadata)} images.")
    print(f"Saved: {INDEX_DIR/'dino.index'}, {INDEX_DIR/'color_hists.npy'}, {INDEX_DIR/'metadata.json'}")


if __name__ == "__main__":
    build()