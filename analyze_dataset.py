"""Analyze COCO dataset: area statistics per category."""
import json
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(r"F:\Bear\apple\dataset\batch1_coco_p03_aligned_crop2")


def analyze_split(split: str) -> dict:
    ann_path = DATASET_ROOT / "annotations" / f"instances_{split}.json"
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    cat_areas = defaultdict(list)
    cat_bbox_ratios = defaultdict(list)  # bbox area / image area

    img_map = {img["id"]: img for img in coco["images"]}

    for ann in coco["annotations"]:
        cat_name = cat_map.get(ann["category_id"], "?")
        area = ann.get("area", 0)
        cat_areas[cat_name].append(area)

        # Compute bbox area ratio
        x, y, w, h = ann["bbox"]
        bbox_area = w * h
        img_info = img_map.get(ann["image_id"])
        if img_info:
            img_area = img_info["width"] * img_info["height"]
            cat_bbox_ratios[cat_name].append(bbox_area / img_area)

    stats = {}
    for cat_name, areas in cat_areas.items():
        if not areas:
            continue
        ratios = cat_bbox_ratios[cat_name]
        stats[cat_name] = {
            "count": len(areas),
            "area_avg": sum(areas) / len(areas),
            "area_min": min(areas),
            "area_max": max(areas),
            "ratio_avg": sum(ratios) / len(ratios) * 100 if ratios else 0,
            "ratio_min": min(ratios) * 100 if ratios else 0,
            "ratio_max": max(ratios) * 100 if ratios else 0,
        }

    return stats, len(coco["images"]), len(coco["annotations"])


def main() -> None:
    lines = []
    lines.append("=" * 60)
    lines.append("COCO Dataset Analysis Report")
    lines.append("=" * 60)
    lines.append(f"Dataset: {DATASET_ROOT.name}")
    lines.append("")

    all_areas = defaultdict(list)
    total_images = 0
    total_anns = 0

    for split in ["train", "val", "test"]:
        stats, n_imgs, n_anns = analyze_split(split)
        total_images += n_imgs
        total_anns += n_anns

        lines.append(f"--- {split.upper()} ---")
        lines.append(f"Images: {n_imgs}, Annotations: {n_anns}")
        for cat_name in sorted(stats.keys()):
            s = stats[cat_name]
            all_areas[cat_name].extend([s["area_min"], s["area_max"]])  # rough; better below
            lines.append(f"  [{cat_name}]")
            lines.append(f"    Count:        {s['count']}")
            lines.append(f"    Area avg:     {s['area_avg']:.2f} px")
            lines.append(f"    Area min:     {s['area_min']:.2f} px")
            lines.append(f"    Area max:     {s['area_max']:.2f} px")
            lines.append(f"    BBox/Img avg: {s['ratio_avg']:.4f}%")
            lines.append(f"    BBox/Img min: {s['ratio_min']:.4f}%")
            lines.append(f"    BBox/Img max: {s['ratio_max']:.4f}%")
        lines.append("")

    # Aggregate stats across all splits
    lines.append("=" * 60)
    lines.append("OVERALL")
    lines.append("=" * 60)
    lines.append(f"Total images: {total_images}")
    lines.append(f"Total annotations: {total_anns}")
    lines.append("")

    for split in ["train", "val", "test"]:
        stats, _, _ = analyze_split(split)
        for cat_name, s in stats.items():
            all_areas[cat_name].extend(
                [a for a in [s["area_avg"], s["area_min"], s["area_max"]]]
            )

    # Actually collect all areas properly
    all_cat_areas = defaultdict(list)
    for split in ["train", "val", "test"]:
        ann_path = DATASET_ROOT / "annotations" / f"instances_{split}.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)
        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        for ann in coco["annotations"]:
            cat_name = cat_map.get(ann["category_id"], "?")
            all_cat_areas[cat_name].append(ann.get("area", 0))

    for cat_name in sorted(all_cat_areas.keys()):
        areas = all_cat_areas[cat_name]
        lines.append(f"[{cat_name}] (all splits combined)")
        lines.append(f"  Total count:  {len(areas)}")
        lines.append(f"  Area avg:     {sum(areas)/len(areas):.2f} px")
        lines.append(f"  Area min:     {min(areas):.2f} px")
        lines.append(f"  Area max:     {max(areas):.2f} px")
        lines.append("")

    report = "\n".join(lines)
    print(report)

    out_path = DATASET_ROOT / "analysis_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to: {out_path}")


if __name__ == "__main__":
    main()
