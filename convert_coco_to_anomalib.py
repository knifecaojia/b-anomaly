"""Convert batch1_coco_p03_aligned_crop2 (COCO detection format)
to anomalib Folder format (train/normal, test/normal, test/abnormal).

Logic:
  - train split: images WITHOUT annotations -> train/normal
  - val split:   images WITHOUT annotations -> test/normal
                 images WITH    annotations -> test/abnormal
  - test split:  images WITHOUT annotations -> test/normal (merged)
                 images WITH    annotations -> test/abnormal (merged)

Usage:
  python convert_coco_to_anomalib.py
  python convert_coco_to_anomalib.py --src dataset/batch1_coco_p03_aligned_crop2 --dst dataset/anomalib_crop2
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from loguru import logger


ROOT = Path(__file__).parent
DEFAULT_SRC = ROOT / "dataset" / "batch1_coco_p03_aligned_crop2"
DEFAULT_DST = ROOT / "dataset" / "anomalib_crop2"


def load_coco(json_path: Path) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def copy_image(src: Path, dst: Path) -> None:
    buf = np.fromfile(str(src), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning(f"Cannot read image: {src}")
        return
    ok, enc = cv2.imencode(src.suffix, img)
    if ok:
        dst.parent.mkdir(parents=True, exist_ok=True)
        enc.tofile(str(dst))


def convert(src_dir: Path, dst_dir: Path) -> None:
    if dst_dir.exists():
        logger.info(f"Removing existing output: {dst_dir}")
        shutil.rmtree(dst_dir)

    stats = defaultdict(int)

    splits_config = [
        ("train", "instances_train.json"),
        ("val",   "instances_val.json"),
        ("test",  "instances_test.json"),
    ]

    for split_name, ann_file in splits_config:
        ann_path = src_dir / "annotations" / ann_file
        img_dir = src_dir / "images" / split_name

        if not ann_path.exists():
            logger.warning(f"Annotation file not found: {ann_path}")
            continue

        coco = load_coco(ann_path)
        img_id_to_info = {img["id"]: img for img in coco["images"]}
        img_id_to_anns = defaultdict(list)
        for ann in coco["annotations"]:
            img_id_to_anns[ann["image_id"]].append(ann)

        for img_id, img_info in img_id_to_info.items():
            fname = img_info["file_name"]
            src_img = img_dir / fname

            if not src_img.exists():
                logger.warning(f"Image not found: {src_img}")
                continue

            has_defect = len(img_id_to_anns.get(img_id, [])) > 0

            if split_name == "train":
                if has_defect:
                    dest_rel = "test/abnormal"
                else:
                    dest_rel = "train/normal"
            else:
                if has_defect:
                    dest_rel = "test/abnormal"
                else:
                    dest_rel = "test/normal"

            dst_img = dst_dir / dest_rel / fname
            copy_image(src_img, dst_img)
            stats[dest_rel] += 1

    logger.info("=" * 50)
    logger.info("COCO -> Anomalib conversion complete")
    logger.info(f"Source: {src_dir}")
    logger.info(f"Output: {dst_dir}")
    logger.info("-" * 50)
    for key in sorted(stats.keys()):
        logger.info(f"  {key}: {stats[key]} images")
    total = sum(stats.values())
    logger.info(f"  Total: {total} images")
    logger.info("=" * 50)

    summary = {
        "source": str(src_dir),
        "output": str(dst_dir),
        "counts": dict(stats),
        "total": total,
    }
    summary_path = dst_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO dataset to anomalib Folder format")
    parser.add_argument("--src", type=str, default=str(DEFAULT_SRC))
    parser.add_argument("--dst", type=str, default=str(DEFAULT_DST))
    args = parser.parse_args()
    convert(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
