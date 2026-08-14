"""
Downloads images listed in the CSV's `image_url` column into data/images/,
named by SKU. Skips already-downloaded files so it's safe to re-run.

Usage:
    python src/download_images.py data/byrappa_tejas_31july.csv
"""

import sys
import csv
import time
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO

IMAGES_DIR = Path("data/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def download_all(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Found {len(rows)} rows in CSV.")

    ok, failed, skipped = 0, 0, 0
    for i, row in enumerate(rows, 1):
        sku = row.get("SKU", "").strip()
        url = row.get("image_url", "").strip()
        if not sku or not url:
            failed += 1
            continue

        out_path = IMAGES_DIR / f"{sku}.jpg"
        if out_path.exists():
            skipped += 1
            continue

        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(out_path, "JPEG", quality=90)
            ok += 1
        except Exception as e:
            print(f"[{i}/{len(rows)}] FAILED {sku}: {e}")
            failed += 1
            continue

        if i % 20 == 0:
            print(f"[{i}/{len(rows)}] downloaded so far: {ok}, failed: {failed}, skipped: {skipped}")

        time.sleep(0.1)

    print(f"\nDone. ok={ok} failed={failed} skipped={skipped} total={len(rows)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/download_images.py <csv_path>")
        sys.exit(1)
    download_all(sys.argv[1])