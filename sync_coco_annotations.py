"""Sync COCO annotations with actual images on disk."""
import json
from pathlib import Path

DATASET_ROOT = Path(r"F:\Bear\apple\dataset\batch1_coco_p03_aligned_crop2")


def sync_split(split: str) -> dict:
    ann_path = DATASET_ROOT / "annotations" / f"instances_{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    img_dir = DATASET_ROOT / "images" / split
    # Find all actual image files on disk
    actual_files = {f.name for f in img_dir.glob("*.jpg")} | {f.name for f in img_dir.glob("*.png")}

    # Filter images that still exist
    new_images = [img for img in coco["images"] if img["file_name"] in actual_files]
    keep_ids = {img["id"] for img in new_images}

    # Filter annotations for kept images
    new_annotations = [ann for ann in coco.get("annotations", []) if ann["image_id"] in keep_ids]

    # Reassign annotation IDs sequentially
    for i, ann in enumerate(new_annotations, 1):
        ann["id"] = i

    new_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    # Stats per category
    cat_counts = {}
    for ann in new_annotations:
        cat_counts[ann["category_id"]] = cat_counts.get(ann["category_id"], 0) + 1

    return {
        "split": split,
        "images": len(new_images),
        "annotations": len(new_annotations),
        "missing": len(coco["images"]) - len(new_images),
        "categories": cat_counts,
    }


def main() -> None:
    print("=" * 60)
    print("COCO Annotation Sync Report")
    print("=" * 60)

    all_stats = []
    for split in ["train", "val", "test"]:
        stats = sync_split(split)
        all_stats.append(stats)
        print(f"\n[{stats['split']}]")
        print(f"  Images:      {stats['images']}")
        print(f"  Annotations: {stats['annotations']}")
        print(f"  Removed:     {stats['missing']} missing images")
        if stats["categories"]:
            print(f"  Categories:  {stats['categories']}")

    total_images = sum(s["images"] for s in all_stats)
    total_anns = sum(s["annotations"] for s in all_stats)
    total_removed = sum(s["missing"] for s in all_stats)

    print("\n" + "=" * 60)
    print(f"Total images:      {total_images}")
    print(f"Total annotations: {total_anns}")
    print(f"Total removed:     {total_removed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
