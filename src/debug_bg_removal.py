"""
Saves the background-removed version of an image so you can eyeball whether
rembg is correctly isolating the saree, or cutting off wrong regions.

Usage:
    python src/debug_bg_removal.py data/images/AA207601.jpg
"""
import sys
from PIL import Image
from embed import remove_background

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/debug_bg_removal.py <image_path>")
        sys.exit(1)

    img = Image.open(sys.argv[1])
    fg_image, alpha_mask = remove_background(img)

    out_path = sys.argv[1].replace(".jpg", "_bgremoved.png")
    fg_image.save(out_path)

    import numpy as np
    fg_pixel_pct = (alpha_mask > 30).sum() / alpha_mask.size * 100
    print(f"Saved: {out_path}")
    print(f"Foreground pixel %: {fg_pixel_pct:.1f}%")