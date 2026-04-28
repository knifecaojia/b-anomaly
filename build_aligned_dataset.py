"""Build aligned dataset: SIFT registration + SAM2 template bbox crop + center.

SAM2 runs only once on the template to get a precise product bbox.
All aligned images are cropped with this same bbox.
"""
import argparse
import logging
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from core.image_registration import ImageRegistration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_raw"
DST_DIR = ROOT / "dataset" / "anomalib_P03_yiwu_aligned"
CHECKPOINT = ROOT / "checkpoints" / "sam2_hiera_small.pt"

random.seed(42)

BG_COLOR = (128, 128, 128)


def build_sam2_auto():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"SAM2 using device: {device}")
    model = build_sam2("sam2_hiera_s.yaml", str(CHECKPOINT), device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.80,
        crop_n_layers=0,
        min_mask_region_area=5000,
    )
    return mask_generator, device


def get_best_mask(mask_generator, image: np.ndarray) -> np.ndarray | None:
    """Auto mask: pick the largest mask that contains image center."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    masks = mask_generator.generate(image)
    if not masks:
        return None

    center_masks = []
    for m in masks:
        seg = m["segmentation"].astype(bool)
        cy, cx = center[1], center[0]
        if 0 <= cy < seg.shape[0] and 0 <= cx < seg.shape[1] and seg[cy, cx]:
            center_masks.append(m)

    if not center_masks:
        center_masks = masks

    best = max(center_masks, key=lambda m: m["area"])
    return best["segmentation"].astype(np.uint8)


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Close holes then dilate to ensure edges are included."""
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    mask = cv2.dilate(mask, kernel_dilate)
    return mask


def get_template_bbox_sam2(
    ref_img: np.ndarray, pad: int = 30, trim_top: int = 80, trim_bottom: int = 80
) -> tuple[int, int, int, int]:
    """Run SAM2 once on template, return (x1, y1, x2, y2) bbox."""
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {CHECKPOINT}")

    logger.info("Loading SAM2 for template bbox extraction...")
    mask_generator, _ = build_sam2_auto()

    logger.info("Generating template mask...")
    mask = get_best_mask(mask_generator, ref_img)
    if mask is None:
        raise RuntimeError("SAM2 failed to produce any mask on template")

    mask = refine_mask(mask)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Empty mask after refinement")

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad + trim_top)
    y2 = min(ref_img.shape[0] - 1, int(ys.max()) + pad - trim_bottom)
    x2 = min(ref_img.shape[1] - 1, int(xs.max()) + pad)
    logger.info(f"Template SAM2 bbox: ({x1},{y1},{x2},{y2}) trim=({trim_top},{trim_bottom})")
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-limit", type=int, default=50,
                        help="Number of training normal images to sample (0 = all).")
    parser.add_argument("--bg-gray", type=int, default=128,
                        help="Background gray level.")
    parser.add_argument("--use-sam2", action="store_true",
                        help="Use SAM2 to compute template bbox (default: use previous Canny+intersection).")
    args = parser.parse_args()

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)

    ref_path = RAW_DIR / "train" / "normal" / "S0000013_P03_20260420101526_20260420101450.jpg"
    ref_img = cv2.imdecode(np.fromfile(str(ref_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None:
        logger.error(f"Failed to load reference: {ref_path}")
        return

    # Get fixed crop bbox
    if args.use_sam2:
        x1, y1, x2, y2 = get_template_bbox_sam2(ref_img)
    else:
        # Fallback: Canny-based detection (original logic from earlier)
        from test_sift_align import detect_product_bbox, content_bbox  # type: ignore
        x1_all, y1_all, x2_all, y2_all = [], [], [], []
        reg_tmp = ImageRegistration(max_features=5000, good_match_percent=0.15,
                                    min_matches=10, method="sift")
        reg_tmp.set_template(ref_img)
        # Just use template itself for fallback bbox
        bbox = detect_product_bbox(ref_img)
        if bbox:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            x1, y1, x2, y2 = 0, 0, ref_img.shape[1], ref_img.shape[0]
        logger.info(f"Fallback Canny bbox: ({x1},{y1},{x2},{y2})")

    fw, fh = x2 - x1 + 1, y2 - y1 + 1
    logger.info(f"Fixed crop size: {fw}x{fh}")

    # Setup registration
    reg = ImageRegistration(max_features=5000, good_match_percent=0.15, min_matches=10, method="sift")
    reg.set_template(ref_img)

    splits = [
        ("train/normal", args.train_limit if args.train_limit > 0 else None),
        ("test/normal", None),
        ("test/abnormal", None),
    ]

    all_files = []
    for split_name, limit in splits:
        src = RAW_DIR / split_name
        if not src.exists():
            continue
        files = sorted(list(src.glob("*.jpg")) + list(src.glob("*.png")))
        if limit:
            random.shuffle(files)
            files = files[:limit]
        all_files.extend((split_name, f) for f in files)

    # Align and crop all images with the same bbox
    logger.info("Aligning and cropping all images...")
    margin = 20
    canvas_w = fw + 2 * margin
    canvas_h = fh + 2 * margin
    bg = (args.bg_gray, args.bg_gray, args.bg_gray)

    for split_name, f in all_files:
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        aligned, matrix, ratio = reg.register(img)
        angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

        h, w = aligned.shape[:2]
        cx1 = max(0, min(x1, w - 1))
        cy1 = max(0, min(y1, h - 1))
        cx2 = max(0, min(x2, w))
        cy2 = max(0, min(y2, h))
        crop = aligned[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]

        canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
        cy = (canvas_h - ch) // 2
        cx = (canvas_w - cw) // 2
        canvas[cy:cy + ch, cx:cx + cw] = crop

        dst_split = DST_DIR / split_name
        dst_split.mkdir(parents=True, exist_ok=True)
        out_path = dst_split / f.name
        ok, buf = cv2.imencode(out_path.suffix, canvas)
        if ok:
            buf.tofile(str(out_path))

        logger.info(f"  {f.name}: angle={angle:+.2f}°, match={ratio:.2f}, out={canvas.shape[1]}x{canvas.shape[0]}")

    logger.info(f"Done. Output: {DST_DIR}")


if __name__ == "__main__":
    main()
