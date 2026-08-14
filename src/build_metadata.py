"""
Builds a JSON lookup from SKU → full product details using the CSV catalog.

Run once (or whenever the CSV changes):
    python src/build_metadata.py

Produces:
    data/index/product_catalog.json
"""

import csv
import json
from pathlib import Path

CSV_PATH = Path("data/byrappa_tejas_31july.csv")
INDEX_DIR = Path("data/index")
OUTPUT_PATH = INDEX_DIR / "product_catalog.json"


def build():
    catalog = {}

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = row.get("SKU", "").strip()
            if not sku:
                continue

            catalog[sku] = {
                "name": row.get("Name", "").strip(),
                "sku": sku,
                "retail_price": row.get("Retail Price", "").strip(),
                "discounted_price": row.get("Discounted Price", "").strip(),
                "image_url": row.get("image_url", "").strip(),
                "website_link": row.get("Website Link", "").strip(),
                "in_stock": int(row.get("Stock", "0").strip() or 0) > 0,
            }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Built product catalog: {len(catalog)} SKUs → {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
