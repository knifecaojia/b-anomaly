"""Extract product from SIFT images and center on uniform background (NO resize).

Pipeline:
  1. Crop SIFT black borders.
  2. Detect product via Canny edges + small padding.
  3. Place product on a UNIFORM canvas at original resolution, centered.
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_sift"
DST_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_centered_noresize"

random.seed(42)


def dynamic_pure_crop(image: np.ndarray, shrink_ratio: float = 0.02) -> np.ndarray:
    """Crop black borders left by SIFT warping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    margin_x = max(1, int(w * shrink_ratio))
    margin_y = max(1, int(h * shrink_ratio))
    return image[y + margin_y:y + h - margin_y, x + margin_x:x + w - margin_x]


def detect_product_bbox(image: np.ndarray, pad: int = 5) -> tuple[int, int, int, int] | None:
    """Detect product bounding box using Canny edges, with padding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coords = cv2.findNonZero(edges)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    # Pad to avoid cutting product edges
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)
    return x, y, w, h


def get_product_size(image: np.ndarray) -> tuple[int, int]:
    """Return (width, height) of detected product region."""
    cropped = dynamic_pure_crop(image)
    bbox = detect_product_bbox(cropped)
    if bbox is None:
        return cropped.shape[1], cropped.shape[0]
    _, _, w, h = bbox
    return w, h


def extract_and_center(
    image: np.ndarray,
    canvas_w: int,
    canvas_h: int,
    bg_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Extract product and place at original resolution on uniform canvas."""
    # 1. Remove black borders
    cropped = dynamic_pure_crop(image)

    # 2. Detect product
    bbox = detect_product_bbox(cropped)
    if bbox is None:
        product = cropped
    else:
        x, y, w, h = bbox
        product = cropped[y:y + h, x:x + w]

    # 3. Place on canvas at original resolution, centered
    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    ph, pw = product.shape[:2]
    cy = (canvas_h - ph) // 2
    cx = (canvas_w - pw) // 2
    canvas[cy:cy + ph, cx:cx + pw] = product
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=50,
                        help="Number of training normal images to sample (0 = all).")
    parser.add_argument("--bg-gray", type=int, default=128,
                        help="Background gray level.")
    args = parser.parse_args()

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    splits = [
        ("train/normal", args.train_limit if args.train_limit > 0 else None),
        ("test/normal", None),
        ("test/abnormal", None),
    ]

    # ── Phase 1: determine max product size across all images ──
    logger.info("[phase 1] scanning all images to find max product size...")
    max_w = max_h = 0
    all_files = []
    for split_name, limit in splits:
        src = SRC_DIR / split_name
        if not src.exists():
            continue
        files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
        if limit:
            random.shuffle(files)
            files = files[:limit]
        all_files.extend((split_name, f) for f in files)

    for split_name, f in all_files:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        w, h = get_product_size(img)
        max_w = max(max_w, w)
        max_h = max(max_h, h)

    # Add margin so no product touches canvas edge
    margin = 20
    canvas_w = max_w + 2 * margin
    canvas_h = max_h + 2 * margin
    logger.info(f"[phase 1] max product size: {max_w}x{max_h}  -> canvas: {canvas_w}x{canvas_h}")

    # ── Phase 2: extract & center each image ──
    bg = (args.bg_gray, args.bg_gray, args.bg_gray)
    for split_name, f in all_files:
        dst_split = DST_DIR / split_name
        dst_split.mkdir(parents=True, exist_ok=True)

        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = extract_and_center(img, canvas_w, canvas_h, bg)
        out_path = dst_split / f.name
        ok, buf = cv2.imencode(out_path.suffix, out)
        if ok:
            buf.tofile(str(out_path))

    logger.info(f"[phase 2] done. Output: {DST_DIR}")


if __name__ == "__main__":
    main()
