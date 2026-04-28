"""Build aligned COCO dataset: SIFT registration + SAM2 template bbox crop + center.

SAM2 runs only once on the template to get a precise product bbox with top/bottom trim.
All aligned images are cropped with this same bbox, and COCO annotations are warped accordingly.
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

from core.image_registration import ImageRegistration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
CHECKPOINT = ROOT / "checkpoints" / "sam2_hiera_small.pt"
SAM2_CONFIG = "sam2_hiera_s.yaml"

BG_GRAY = 128
MARGIN = 20


def build_sam2_auto():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"SAM2 using device: {device}")
    model = build_sam2(SAM2_CONFIG, str(CHECKPOINT), device=device)
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
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    mask = cv2.dilate(mask, kernel_dilate)
    return mask


def get_template_bbox_sam2(
    ref_img: np.ndarray, pad: int = 30, trim_top: int = 80, trim_bottom: int = 80
) -> tuple[int, int, int, int]:
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


def transform_bbox(bbox: list[float], matrix: np.ndarray) -> list[float]:
    """Transform COCO bbox [x, y, w, h] through 2x3 affine matrix."""
    x, y, w, h = bbox
    pts = np.array([
        [x, y], [x + w, y], [x, y + h], [x + w, y + h]
    ], dtype=np.float32)
    pts_h = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)
    transformed = (matrix @ pts_h.T).T
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    x1, y1 = float(xs.min()), float(ys.min())
    x2, y2 = float(xs.max()), float(ys.max())
    return [x1, y1, x2 - x1, y2 - y1]


def warp_annotation(ann: dict, matrix: np.ndarray, crop_x1: int, crop_y1: int, margin: int) -> dict | None:
    """Warp a single COCO annotation and shift by crop box + margin.
    Returns None if bbox becomes invalid after crop.
    """
    new_ann = dict(ann)

    # Transform bbox
    if "bbox" in ann:
        new_bbox = transform_bbox(ann["bbox"], matrix)
        # Shift by crop origin and margin
        new_bbox[0] = new_bbox[0] - crop_x1 + margin
        new_bbox[1] = new_bbox[1] - crop_y1 + margin
        # Clamp negatives
        if new_bbox[2] <= 0 or new_bbox[3] <= 0:
            return None
        if new_bbox[0] + new_bbox[2] <= 0 or new_bbox[1] + new_bbox[3] <= 0:
            return None
        new_ann["bbox"] = [round(v, 2) for v in new_bbox]

    # Recompute area from new bbox if no segmentation
    if "area" in new_ann and not new_ann.get("segmentation"):
        new_ann["area"] = round(new_bbox[0] * new_bbox[1], 2) if "new_bbox" in dir() else new_ann["area"]
        # Actually area = w * h
        new_ann["area"] = round(new_bbox[2] * new_bbox[3], 2)

    # Transform segmentation polygons if present
    if "segmentation" in ann and ann["segmentation"]:
        new_segs = []
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
            transformed = (matrix @ pts_h.T).T
            transformed[:, 0] = transformed[:, 0] - crop_x1 + margin
            transformed[:, 1] = transformed[:, 1] - crop_y1 + margin
            new_segs.append(transformed.flatten().tolist())
        new_ann["segmentation"] = new_segs

    return new_ann


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    reg: ImageRegistration,
    x1: int, y1: int, x2: int, y2: int,
    bg: tuple[int, int, int],
) -> None:
    ann_path = src_root / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        logger.warning(f"Annotation not found: {ann_path}")
        return

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco.get("annotations", [])

    # Build lookups
    img_dir = src_root / "images" / split
    dst_img_dir = dst_root / "images" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    fw, fh = x2 - x1 + 1, y2 - y1 + 1
    canvas_w = fw + 2 * MARGIN
    canvas_h = fh + 2 * MARGIN

    # Group annotations by image_id
    ann_by_img: dict[int, list[dict]] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    ann_id_counter = 1

    for img_info in images:
        file_name = img_info["file_name"]
        src_path = img_dir / file_name
        if not src_path.exists():
            logger.warning(f"Image not found: {src_path}")
            continue

        img = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning(f"Failed to read image: {src_path}")
            continue

        # SIFT align
        aligned, matrix, ratio = reg.register(img)
        angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

        # Crop with fixed bbox
        h, w = aligned.shape[:2]
        cx1 = max(0, min(x1, w - 1))
        cy1 = max(0, min(y1, h - 1))
        cx2 = max(0, min(x2, w))
        cy2 = max(0, min(y2, h))
        crop = aligned[cy1:cy2, cx1:cx2]
        ch, cw = crop.shape[:2]

        # Center on canvas
        canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
        off_y = (canvas_h - ch) // 2
        off_x = (canvas_w - cw) // 2
        canvas[off_y:off_y + ch, off_x:off_x + cw] = crop

        # Save aligned image
        dst_path = dst_img_dir / file_name
        ok, buf = cv2.imencode(dst_path.suffix, canvas)
        if ok:
            buf.tofile(str(dst_path))

        # Update image info
        new_img_info = dict(img_info)
        new_img_info["width"] = canvas_w
        new_img_info["height"] = canvas_h
        # Keep original file_name
        new_images.append(new_img_info)

        # Warp annotations
        for ann in ann_by_img.get(img_info["id"], []):
            new_ann = warp_annotation(ann, matrix, cx1, cy1, MARGIN)
            if new_ann is not None:
                new_ann["id"] = ann_id_counter
                ann_id_counter += 1
                new_ann["image_id"] = img_info["id"]
                new_annotations.append(new_ann)

        logger.info(f"  {file_name}: angle={angle:+.2f}°, match={ratio:.2f}, out={canvas_w}x{canvas_h}")

    # Save new COCO annotation
    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }
    dst_ann_dir = dst_root / "annotations"
    dst_ann_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_ann_dir / f"instances_{split}.json", "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    logger.info(f"Split {split}: {len(new_images)} images, {len(new_annotations)} annotations")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(ROOT / "dataset" / "batch1_coco_p03"))
    parser.add_argument("--dst", type=str, default=str(ROOT / "dataset" / "batch1_coco_p03_aligned"))
    parser.add_argument("--ref", type=str, default=None,
                        help="Reference image filename for SIFT template. Default: first image in train.")
    parser.add_argument("--trim-top", type=int, default=80)
    parser.add_argument("--trim-bottom", type=int, default=80)
    parser.add_argument("--pad", type=int, default=30)
    parser.add_argument("--bg-gray", type=int, default=128)
    parser.add_argument("--skip-sam2", action="store_true",
                        help="Skip SAM2 and use full image as template bbox.")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Select reference image
    train_img_dir = src_root / "images" / "train"
    if args.ref:
        ref_path = train_img_dir / args.ref
    else:
        candidates = sorted(list(train_img_dir.glob("*.jpg")) + list(train_img_dir.glob("*.png")))
        if not candidates:
            raise FileNotFoundError(f"No images found in {train_img_dir}")
        ref_path = candidates[0]

    logger.info(f"Reference template: {ref_path.name}")
    ref_img = cv2.imdecode(np.fromfile(str(ref_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None:
        raise RuntimeError(f"Failed to load reference image: {ref_path}")

    # Get template bbox via SAM2
    if args.skip_sam2:
        x1, y1, x2, y2 = 0, 0, ref_img.shape[1] - 1, ref_img.shape[0] - 1
        logger.info(f"Using full image bbox: ({x1},{y1},{x2},{y2})")
    else:
        x1, y1, x2, y2 = get_template_bbox_sam2(ref_img, pad=args.pad, trim_top=args.trim_top, trim_bottom=args.trim_bottom)

    # Setup SIFT registration
    reg = ImageRegistration(max_features=5000, good_match_percent=0.15, min_matches=10, method="sift")
    reg.set_template(ref_img)

    bg = (args.bg_gray, args.bg_gray, args.bg_gray)

    for split in ["train", "val", "test"]:
        logger.info(f"Processing split: {split}")
        process_split(src_root, dst_root, split, reg, x1, y1, x2, y2, bg)

    logger.info(f"Done. Aligned COCO dataset at: {dst_root}")


if __name__ == "__main__":
    main()
