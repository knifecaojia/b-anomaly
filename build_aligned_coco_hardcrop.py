"""Build aligned COCO dataset: SIFT registration + hard crop (top 100px, bottom 150px).

No SAM2. No smart detection. Straight hard crop after SIFT alignment.
Reference image: S0000015_P03_20260420101540_20260420101508.jpg
"""
import argparse
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from core.image_registration import ImageRegistration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
MARGIN = 20

# Hard crop params
TRIM_TOP = 100
TRIM_BOTTOM = 150


def transform_bbox(bbox: list[float], matrix: np.ndarray) -> list[float]:
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


def warp_annotation(ann: dict, matrix: np.ndarray, crop_y1: int, margin: int, canvas_w: int, canvas_h: int) -> dict | None:
    new_ann = dict(ann)

    if "bbox" in ann:
        new_bbox = transform_bbox(ann["bbox"], matrix)
        new_bbox[0] = new_bbox[0] + margin
        new_bbox[1] = new_bbox[1] - crop_y1 + margin
        if new_bbox[2] <= 0 or new_bbox[3] <= 0:
            return None
        if new_bbox[0] + new_bbox[2] <= 0 or new_bbox[1] + new_bbox[3] <= 0:
            return None
        if new_bbox[0] >= canvas_w or new_bbox[1] >= canvas_h:
            return None
        # Clamp to canvas
        new_bbox[0] = max(0.0, new_bbox[0])
        new_bbox[1] = max(0.0, new_bbox[1])
        new_bbox[2] = min(new_bbox[2], canvas_w - new_bbox[0])
        new_bbox[3] = min(new_bbox[3], canvas_h - new_bbox[1])
        new_ann["bbox"] = [round(v, 2) for v in new_bbox]
        new_ann["area"] = round(new_bbox[2] * new_bbox[3], 2)

    if "segmentation" in ann and ann["segmentation"]:
        new_segs = []
        for poly in ann["segmentation"]:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
            transformed = (matrix @ pts_h.T).T
            transformed[:, 0] = transformed[:, 0] + margin
            transformed[:, 1] = transformed[:, 1] - crop_y1 + margin
            # Simple check: all points inside canvas
            if np.any(transformed[:, 0] < 0) or np.any(transformed[:, 1] < 0):
                continue
            if np.any(transformed[:, 0] >= canvas_w) or np.any(transformed[:, 1] >= canvas_h):
                continue
            new_segs.append(transformed.flatten().tolist())
        if not new_segs:
            return None
        new_ann["segmentation"] = new_segs

    return new_ann


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    reg: ImageRegistration,
    trim_top: int,
    trim_bottom: int,
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

    img_dir = src_root / "images" / split
    dst_img_dir = dst_root / "images" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    # Template size determines aligned size
    template_h, template_w = reg.template.shape[:2]
    crop_y1 = trim_top
    crop_y2 = template_h - trim_bottom
    crop_h = crop_y2 - crop_y1
    crop_w = template_w

    canvas_w = crop_w + 2 * MARGIN
    canvas_h = crop_h + 2 * MARGIN

    ann_by_img: dict[int, list[dict]] = {}
    for ann in annotations:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    new_images = []
    new_annotations = []
    ann_id_counter = 1
    kept = 0

    for img_info in images:
        file_name = img_info["file_name"]
        src_path = img_dir / file_name
        if not src_path.exists():
            continue

        img = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        aligned, matrix, ratio = reg.register(img)
        angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))

        # Hard crop: top trim_top, bottom trim_bottom
        h, w = aligned.shape[:2]
        cy1 = max(0, min(crop_y1, h))
        cy2 = max(0, min(crop_y2, h))
        crop = aligned[cy1:cy2, :]
        ch, cw = crop.shape[:2]

        # Center on canvas
        canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
        off_y = (canvas_h - ch) // 2
        off_x = (canvas_w - cw) // 2
        canvas[off_y:off_y + ch, off_x:off_x + cw] = crop

        # Save
        dst_path = dst_img_dir / file_name
        ok, buf = cv2.imencode(dst_path.suffix, canvas)
        if ok:
            buf.tofile(str(dst_path))

        new_img_info = dict(img_info)
        new_img_info["width"] = canvas_w
        new_img_info["height"] = canvas_h
        new_images.append(new_img_info)

        for ann in ann_by_img.get(img_info["id"], []):
            new_ann = warp_annotation(ann, matrix, cy1, MARGIN, canvas_w, canvas_h)
            if new_ann is not None:
                new_ann["id"] = ann_id_counter
                ann_id_counter += 1
                new_ann["image_id"] = img_info["id"]
                new_annotations.append(new_ann)

        kept += 1
        if kept % 50 == 0:
            logger.info(f"  ... {split} processed {kept}/{len(images)}")

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

    logger.info(f"Split {split}: kept={kept}, annotations={len(new_annotations)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=str(ROOT / "dataset" / "batch1_coco_p03"))
    parser.add_argument("--dst", type=str, default=str(ROOT / "dataset" / "batch1_coco_p03_aligned"))
    parser.add_argument("--ref", type=str, default="S0000015_P03_20260420101540_20260420101508.jpg")
    parser.add_argument("--trim-top", type=int, default=TRIM_TOP)
    parser.add_argument("--trim-bottom", type=int, default=TRIM_BOTTOM)
    parser.add_argument("--bg-gray", type=int, default=128)
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    ref_path = src_root / "images" / "train" / args.ref
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference not found: {ref_path}")

    ref_img = cv2.imdecode(np.fromfile(str(ref_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref_img is None:
        raise RuntimeError(f"Failed to load reference: {ref_path}")

    logger.info(f"Template: {ref_path.name} {ref_img.shape[1]}x{ref_img.shape[0]}")
    logger.info(f"Hard crop: top={args.trim_top}, bottom={args.trim_bottom}")

    reg = ImageRegistration(max_features=5000, good_match_percent=0.15, min_matches=10, method="sift")
    reg.set_template(ref_img)

    bg = (args.bg_gray, args.bg_gray, args.bg_gray)

    for split in ["train", "val", "test"]:
        logger.info(f"Processing: {split}")
        process_split(src_root, dst_root, split, reg, args.trim_top, args.trim_bottom, bg)

    logger.info(f"Done. Output: {dst_root}")


if __name__ == "__main__":
    main()
