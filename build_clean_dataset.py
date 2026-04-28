"""Copy clean images (no bottom black border) from aligned dataset to new folder."""
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
SRC_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_aligned"
DST_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_aligned_clean"


def has_bottom_black_border(image: np.ndarray, rows: int = 100, threshold: int = 30, ratio: float = 0.7) -> bool:
    h, w = image.shape[:2]
    if h <= rows:
        bottom = image
    else:
        bottom = image[-rows:, :]
    gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
    black_mask = gray < threshold
    black_ratio = black_mask.sum() / black_mask.size
    return black_ratio > ratio


def main():
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(SRC_DIR.rglob("*.jpg")) + sorted(SRC_DIR.rglob("*.png"))
    kept = 0
    removed = 0

    for f in image_files:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        if has_bottom_black_border(img):
            removed += 1
            logger.info(f"REMOVE (bottom border): {f.relative_to(SRC_DIR)}")
            continue

        # Copy to new folder preserving relative path
        rel_path = f.relative_to(SRC_DIR)
        dst_path = DST_DIR / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst_path)
        kept += 1

    logger.info(f"---")
    logger.info(f"Total: {len(image_files)}, Kept: {kept}, Removed: {removed}")
    logger.info(f"Clean dataset: {DST_DIR}")


if __name__ == "__main__":
    main()
