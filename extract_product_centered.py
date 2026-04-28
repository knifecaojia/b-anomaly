"""Extract product from SIFT images and center on uniform background.

This removes position drift by cropping to the detected product region,
then resizing so the product fills the entire output resolution.
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
DST_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_centered"

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


def detect_product_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Detect product bounding box using Canny edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coords = cv2.findNonZero(edges)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, w, h


def extract_and_resize(
    image: np.ndarray,
    target_size: int,
    bg_color: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Extract product, center on background, resize to target_size x target_size."""
    # 1. Remove black borders
    cropped = dynamic_pure_crop(image)

    # 2. Detect product
    bbox = detect_product_bbox(cropped)
    if bbox is None:
        # Fallback: use whole cropped image
        product = cropped
    else:
        x, y, w, h = bbox
        # Small margin to avoid cutting edges
        margin = 2
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(cropped.shape[1] - x, w + 2 * margin)
        h = min(cropped.shape[0] - y, h + 2 * margin)
        product = cropped[y:y + h, x:x + w]

    # 3. Resize product to target size (fills entire frame)
    resized = cv2.resize(product, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized


def process_split(src_split: Path, dst_split: Path, target_size: int, file_limit: int | None = None) -> int:
    """Process all images in a split."""
    if dst_split.exists():
        shutil.rmtree(dst_split)
    dst_split.mkdir(parents=True, exist_ok=True)

    files = sorted(list(src_split.glob("*.jpg")) + list(src_split.glob("*.png")))
    if file_limit:
        random.shuffle(files)
        files = files[:file_limit]

    logger.info(f"[process] {src_split.name}: {len(files)} files -> {target_size}x{target_size}")
    kept = 0
    for idx, f in enumerate(files):
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        out = extract_and_resize(img, target_size)
        out_path = dst_split / f.name
        ok, buf = cv2.imencode(out_path.suffix, out)
        if ok:
            buf.tofile(str(out_path))
            kept += 1
        if (idx + 1) % 25 == 0:
            logger.info(f"  {idx + 1}/{len(files)}")
    logger.info(f"  done: kept {kept} images")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=768,
                        help="Output resolution (must be divisible by 64). Default 768.")
    parser.add_argument("--train-limit", type=int, default=50,
                        help="Number of training normal images to sample (0 = all).")
    parser.add_argument("--bg-gray", type=int, default=128,
                        help="Background gray level if padding is needed.")
    args = parser.parse_args()

    if args.image_size % 64 != 0:
        raise SystemExit(f"--image-size must be divisible by 64 (got {args.image_size})")

    splits = [
        ("train/normal", args.train_limit if args.train_limit > 0 else None),
        ("test/normal", None),
        ("test/abnormal", None),
    ]

    for split_name, limit in splits:
        src = SRC_DIR / split_name
        dst = DST_DIR / split_name
        if not src.exists():
            logger.warning(f"Source not found: {src}")
            continue
        process_split(src, dst, args.image_size, limit)

    logger.info(f"Done. Output: {DST_DIR}")


if __name__ == "__main__":
    main()
