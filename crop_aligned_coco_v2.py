"""Hard crop aligned COCO dataset v2.

Crops from batch1_coco_p03_aligned_crop:
- left: 100px
- right: 500px
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

SRC = Path(r"F:\Bear\apple\dataset\batch1_coco_p03_aligned_crop")
DST = Path(r"F:\Bear\apple\dataset\batch1_coco_p03_aligned_crop2")

LEFT = 100
RIGHT = 500
TOP = 0
BOTTOM = 0


def process_split(split: str) -> None:
    ann_path = SRC / "annotations" / f"instances_{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    src_img_dir = SRC / "images" / split
    dst_img_dir = DST / "images" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    orig_w = coco["images"][0]["width"]
    orig_h = coco["images"][0]["height"]
    new_w = orig_w - LEFT - RIGHT
    new_h = orig_h - TOP - BOTTOM
    print(f"[{split}] {orig_w}x{orig_h} -> {new_w}x{new_h}")

    new_images = []
    new_annotations = []
    ann_id = 1

    for img_info in coco["images"]:
        file_name = img_info["file_name"]
        img = cv2.imdecode(np.fromfile(str(src_img_dir / file_name), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        crop = img[TOP:new_h + TOP, LEFT:new_w + LEFT]
        ok, buf = cv2.imencode(Path(file_name).suffix, crop)
        if ok:
            buf.tofile(str(dst_img_dir / file_name))

        new_img = dict(img_info)
        new_img["width"] = new_w
        new_img["height"] = new_h
        new_images.append(new_img)

        for ann in coco.get("annotations", []):
            if ann["image_id"] != img_info["id"]:
                continue

            new_ann = dict(ann)
            x, y, w, h = ann["bbox"]
            x -= LEFT
            y -= TOP

            if x + w <= 0 or y + h <= 0 or x >= new_w or y >= new_h:
                continue

            x = max(0.0, x)
            y = max(0.0, y)
            w = min(w, new_w - x)
            h = min(h, new_h - y)
            if w <= 0 or h <= 0:
                continue

            new_ann["bbox"] = [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]
            new_ann["area"] = round(w * h, 2)
            new_ann["id"] = ann_id
            ann_id += 1
            new_annotations.append(new_ann)

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }
    dst_ann_dir = DST / "annotations"
    dst_ann_dir.mkdir(parents=True, exist_ok=True)
    with open(dst_ann_dir / f"instances_{split}.json", "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    print(f"[{split}] kept={len(new_images)}, annotations={len(new_annotations)}")


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        process_split(split)

    print(f"Done. Output: {DST}")


if __name__ == "__main__":
    main()
