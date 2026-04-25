from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger


class DatasetManager:
    def __init__(self, base_dir: str | Path = "dataset"):
        self.base_dir = Path(base_dir)

    def get_coco_stats(self, annotation_path: str | Path) -> dict:
        with open(annotation_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        label_counts = Counter()
        for ann in coco["annotations"]:
            name = cat_map.get(ann["category_id"], "unknown")
            label_counts[name] += 1

        return {
            "total_images": len(coco["images"]),
            "total_annotations": len(coco["annotations"]),
            "categories": dict(label_counts),
            "category_names": list(cat_map.values()),
        }

    def merge_coco_datasets(
        self,
        sources: List[Tuple[str | Path, str | Path]],
        output_dir: str | Path,
        class_merge_map: Optional[Dict[str, str]] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> str:
        output_dir = Path(output_dir)
        all_images: List[dict] = []
        all_annotations: List[dict] = []
        all_categories: List[dict] = []
        img_offset = 0
        ann_offset = 0
        cat_id_offset = 0

        for image_dir, ann_path in sources:
            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)

            cat_name_to_new_id: Dict[int, int] = {}
            for cat in coco["categories"]:
                raw_name = cat["name"]
                merged_name = (
                    class_merge_map.get(raw_name, raw_name) if class_merge_map else raw_name
                )
                existing = next(
                    (c for c in all_categories if c["name"] == merged_name), None
                )
                if existing:
                    cat_name_to_new_id[cat["id"]] = existing["id"]
                else:
                    new_id = cat_id_offset + 1
                    all_categories.append({"id": new_id, "name": merged_name})
                    cat_name_to_new_id[cat["id"]] = new_id
                    cat_id_offset = new_id

            img_dir = Path(image_dir)
            for img_info in coco["images"]:
                old_id = img_info["id"]
                new_img = dict(img_info)
                new_img["id"] = img_offset + 1
                new_img["file_name"] = str(img_dir / img_info["file_name"])
                new_img["_old_id"] = old_id
                all_images.append(new_img)

                for ann in coco["annotations"]:
                    if ann["image_id"] == old_id:
                        new_ann = dict(ann)
                        new_ann["id"] = ann_offset + 1
                        new_ann["image_id"] = new_img["id"]
                        new_ann["category_id"] = cat_name_to_new_id[ann["category_id"]]
                        all_annotations.append(new_ann)
                        ann_offset += 1

                img_offset = new_img["id"]

            img_offset = len(all_images)
            ann_offset = len(all_annotations)

        random.seed(seed)
        random.shuffle(all_images)
        img_id_map = {img["_old_id"] if "_old_id" in img else img["id"]: img["id"] for img in all_images}

        n = len(all_images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            "train": all_images[:n_train],
            "val": all_images[n_train : n_train + n_val],
            "test": all_images[n_train + n_val :],
        }

        for split_name, split_imgs in splits.items():
            split_img_ids = {img["id"] for img in split_imgs}
            split_anns = [a for a in all_annotations if a["image_id"] in split_img_ids]

            split_coco = {
                "images": [
                    {k: v for k, v in img.items() if not k.startswith("_")}
                    for img in split_imgs
                ],
                "annotations": split_anns,
                "categories": all_categories,
            }

            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            with open(split_dir / "annotations.json", "w", encoding="utf-8") as f:
                json.dump(split_coco, f, ensure_ascii=False, indent=2)

            logger.info(
                f"{split_name}: {len(split_imgs)} 张图片, {len(split_anns)} 个标注"
            )

        logger.info(f"数据集合并完成，输出到: {output_dir}")
        return str(output_dir)

    def _find_ann_path(self, coco_dir: Path, split: str) -> Optional[Path]:
        candidates = [
            coco_dir / "annotations" / f"instances_{split}.json",
            coco_dir / split / "annotations.json",
            coco_dir / split / "annotations" / "instances.json",
            coco_dir / "annotations" / f"{split}.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _find_img_dir(self, coco_dir: Path, split: str) -> Optional[Path]:
        candidates = [
            coco_dir / "images" / split,
            coco_dir / split / "images",
            coco_dir / "images",
        ]
        for p in candidates:
            if p.exists() and p.is_dir():
                return p
        return None

    def prepare_yolo_dataset(
        self,
        coco_dir: str | Path,
        output_dir: str | Path,
    ) -> str:
        import yaml

        coco_dir = Path(coco_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
        }

        class_names: List[str] = []
        for split in ["train", "val", "test"]:
            ann_path = self._find_ann_path(coco_dir, split)
            if ann_path:
                with open(ann_path, "r", encoding="utf-8") as f:
                    coco = json.load(f)
                if not class_names and coco["categories"]:
                    cat_sorted = sorted(coco["categories"], key=lambda x: x["id"])
                    class_names = [c["name"] for c in cat_sorted]

        data["names"] = {i: n for i, n in enumerate(class_names)}
        data["nc"] = len(class_names)

        for split in ["train", "val", "test"]:
            ann_path = self._find_ann_path(coco_dir, split)
            if not ann_path:
                continue

            with open(ann_path, "r", encoding="utf-8") as f:
                coco = json.load(f)

            img_src_dir = self._find_img_dir(coco_dir, split)
            if img_src_dir is None:
                logger.warning(f"找不到 {split} 的图片目录")
                continue

            img_out = output_dir / split / "images"
            lbl_out = output_dir / split / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            img_id_to_anns: Dict[int, List[dict]] = {}
            for ann in coco["annotations"]:
                img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

            cat_id_to_idx = {c["id"]: i for i, c in enumerate(
                sorted(coco["categories"], key=lambda x: x["id"])
            )}

            for img_info in coco["images"]:
                fname = img_info["file_name"]
                src = img_src_dir / fname
                dst = img_out / Path(fname).name
                if not dst.exists():
                    try:
                        shutil.copy2(str(src), str(dst))
                    except Exception as e:
                        logger.warning(f"复制图片失败 {src}: {e}")
                        continue

                w, h = img_info["width"], img_info["height"]
                lines: List[str] = []
                for ann in img_id_to_anns.get(img_info["id"], []):
                    bbox = ann["bbox"]
                    cx = (bbox[0] + bbox[2] / 2) / w
                    cy = (bbox[1] + bbox[3] / 2) / h
                    bw = bbox[2] / w
                    bh = bbox[3] / h
                    cls_idx = cat_id_to_idx.get(ann["category_id"], 0)
                    lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                lbl_file = lbl_out / (Path(fname).stem + ".txt")
                lbl_file.write_text("\n".join(lines), encoding="utf-8")

            logger.info(f"{split}: {len(coco['images'])} 张图片已转换")

        yaml_path = output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"YOLO 数据集准备完成: {output_dir}")
        return str(yaml_path)
