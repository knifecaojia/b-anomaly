import json
import os
import shutil
from pathlib import Path


def filter_p03_coco(src_root: str, dst_root: str):
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    splits = ["train", "val", "test"]

    for split in splits:
        ann_path = src_root / "annotations" / f"instances_{split}.json"
        if not ann_path.exists():
            print(f"Skip missing: {ann_path}")
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # 过滤含 P03 的图片
        keep_images = [img for img in coco["images"] if "P03" in img.get("file_name", "")]
        keep_ids = {img["id"] for img in keep_images}

        if not keep_images:
            print(f"Split {split}: no P03 images, skip.")
            continue

        # 过滤 annotations
        keep_annotations = [ann for ann in coco.get("annotations", []) if ann["image_id"] in keep_ids]

        # 构建新的 COCO 结构
        new_coco = {
            "info": coco.get("info", {}),
            "licenses": coco.get("licenses", []),
            "images": keep_images,
            "annotations": keep_annotations,
            "categories": coco.get("categories", []),
        }

        # 保存新标注
        dst_ann_dir = dst_root / "annotations"
        dst_ann_dir.mkdir(parents=True, exist_ok=True)
        dst_ann_path = dst_ann_dir / f"instances_{split}.json"
        with open(dst_ann_path, "w", encoding="utf-8") as f:
            json.dump(new_coco, f, ensure_ascii=False, indent=2)

        # 复制图片（保持 images/{split} 结构）
        dst_img_dir = dst_root / "images" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = src_root / "images" / split
        copied = 0
        for img in keep_images:
            src_img = src_img_dir / img["file_name"]
            dst_img = dst_img_dir / img["file_name"]
            if src_img.exists():
                shutil.copy2(src_img, dst_img)
                copied += 1
            else:
                # 尝试全局搜索图片（可能不在对应split目录）
                found = False
                for sp in splits:
                    alt = src_root / "images" / sp / img["file_name"]
                    if alt.exists():
                        shutil.copy2(alt, dst_img)
                        copied += 1
                        found = True
                        break
                if not found:
                    print(f"  Warning: image not found {img['file_name']}")

        print(f"Split {split}: {len(keep_images)} images, {len(keep_annotations)} annotations, {copied} copied.")

    print(f"\nDone. Output at: {dst_root}")


if __name__ == "__main__":
    filter_p03_coco(
        src_root=r"F:\Bear\apple\dataset\batch1_coco",
        dst_root=r"F:\Bear\apple\dataset\batch1_coco_p03",
    )
