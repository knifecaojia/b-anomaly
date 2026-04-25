from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class SliceCoord:
    x_start: int
    y_start: int
    width: int
    height: int


@dataclass
class Annotation:
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int = -1


@dataclass
class SliceResult:
    image: np.ndarray
    annotations: List[Annotation]
    coord: SliceCoord
    has_defect: bool


def compute_slice_coords(
    img_w: int,
    img_h: int,
    slice_size: int = 640,
    overlap: float = 0.25,
) -> List[SliceCoord]:
    if overlap >= 1.0:
        raise ValueError(f"overlap 必须 < 1.0, 当前值: {overlap}")
    stride = int(slice_size * (1 - overlap))
    if stride <= 0:
        stride = 1

    coords: List[SliceCoord] = []
    y = 0
    while y < img_h:
        x = 0
        while x < img_w:
            sw = min(slice_size, img_w - x)
            sh = min(slice_size, img_h - y)
            coords.append(SliceCoord(x_start=x, y_start=y, width=sw, height=sh))
            x += stride
            if x + slice_size > img_w and x < img_w:
                break
        y += stride
        if y + slice_size > img_h and y < img_h:
            break

    return coords


def clip_annotation_to_slice(
    ann: Annotation,
    slice_coord: SliceCoord,
    min_area_ratio: float = 0.3,
) -> Optional[Annotation]:
    sx1 = max(ann.x1, slice_coord.x_start)
    sy1 = max(ann.y1, slice_coord.y_start)
    sx2 = min(ann.x2, slice_coord.x_start + slice_coord.width)
    sy2 = min(ann.y2, slice_coord.y_start + slice_coord.height)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    orig_area = (ann.x2 - ann.x1) * (ann.y2 - ann.y1)
    clipped_area = (sx2 - sx1) * (sy2 - sy1)

    if orig_area > 0 and clipped_area / orig_area < min_area_ratio:
        return None

    local_x1 = sx1 - slice_coord.x_start
    local_y1 = sy1 - slice_coord.y_start
    local_x2 = sx2 - slice_coord.x_start
    local_y2 = sy2 - slice_coord.y_start

    return Annotation(
        x1=local_x1,
        y1=local_y1,
        x2=local_x2,
        y2=local_y2,
        class_name=ann.class_name,
        class_id=ann.class_id,
    )


def slice_image_with_annotations(
    image: np.ndarray,
    annotations: List[Annotation],
    slice_size: int = 640,
    overlap: float = 0.25,
    min_area_ratio: float = 0.3,
) -> List[SliceResult]:
    img_h, img_w = image.shape[:2]
    coords = compute_slice_coords(img_w, img_h, slice_size, overlap)
    results: List[SliceResult] = []

    for coord in coords:
        crop = image[
            coord.y_start : coord.y_start + coord.height,
            coord.x_start : coord.x_start + coord.width,
        ]

        if coord.width < slice_size or coord.height < slice_size:
            padded = np.zeros((slice_size, slice_size, 3), dtype=image.dtype)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded

        slice_anns: List[Annotation] = []
        for ann in annotations:
            clipped = clip_annotation_to_slice(ann, coord, min_area_ratio)
            if clipped is not None:
                slice_anns.append(clipped)

        results.append(
            SliceResult(
                image=crop,
                annotations=slice_anns,
                coord=coord,
                has_defect=len(slice_anns) > 0,
            )
        )

    return results


def slice_image_inference(
    image: np.ndarray,
    slice_size: int = 640,
    overlap: float = 0.25,
) -> List[Tuple[np.ndarray, SliceCoord]]:
    img_h, img_w = image.shape[:2]
    if min(img_w, img_h) <= 2 * slice_size:
        return [(image, SliceCoord(0, 0, img_w, img_h))]

    coords = compute_slice_coords(img_w, img_h, slice_size, overlap)
    slices: List[Tuple[np.ndarray, SliceCoord]] = []

    for coord in coords:
        crop = image[
            coord.y_start : coord.y_start + coord.height,
            coord.x_start : coord.x_start + coord.width,
        ]
        if coord.width < slice_size or coord.height < slice_size:
            padded = np.zeros((slice_size, slice_size, 3), dtype=image.dtype)
            padded[: crop.shape[0], : crop.shape[1]] = crop
            crop = padded
        slices.append((crop, coord))

    return slices


def map_detections_to_global(
    local_boxes: np.ndarray,
    slice_coord: SliceCoord,
) -> np.ndarray:
    if len(local_boxes) == 0:
        return local_boxes

    global_boxes = local_boxes.copy()
    global_boxes[:, 0] += slice_coord.x_start
    global_boxes[:, 1] += slice_coord.y_start
    global_boxes[:, 2] += slice_coord.x_start
    global_boxes[:, 3] += slice_coord.y_start
    return global_boxes


def nms_across_slices(
    all_boxes: np.ndarray,
    all_scores: np.ndarray,
    all_class_ids: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(all_boxes) == 0:
        return all_boxes, all_scores, all_class_ids

    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = all_scores.argsort()[::-1]
    keep: List[int] = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    keep_arr = np.array(keep)
    return all_boxes[keep_arr], all_scores[keep_arr], all_class_ids[keep_arr]


def generate_training_slices(
    image_dir: str | Path,
    coco_annotation_path: str | Path,
    output_dir: str | Path,
    slice_size: int = 640,
    overlap: float = 0.25,
    min_area_ratio: float = 0.3,
    negative_sample_ratio: float = 0.2,
    class_merge_map: Optional[Dict[str, str]] = None,
    split: str = "train",
) -> str:
    import json

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    split_img_dir = output_dir / "images" / split
    split_img_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name: Dict[int, str] = {}
    for cat in coco["categories"]:
        cat_id_to_name[cat["id"]] = cat["name"]

    img_id_to_info: Dict[int, dict] = {}
    for img_info in coco["images"]:
        img_id_to_info[img_info["id"]] = img_info

    img_id_to_anns: Dict[int, List[dict]] = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    target_classes = sorted(
        set(
            class_merge_map.get(cat_id_to_name[cid], cat_id_to_name[cid])
            for cid in cat_id_to_name
        )
    )
    class_to_id = {name: idx for idx, name in enumerate(target_classes)}

    new_images: List[dict] = []
    new_annotations: List[dict] = []
    ann_id_counter = 1
    img_id_counter = 1
    total_defect_slices = 0
    total_negative_slices = 0

    for img_id, img_info in img_id_to_info.items():
        img_path = image_dir / img_info["file_name"]
        if not img_path.exists():
            logger.warning(f"图片不存在: {img_path}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        annotations: List[Annotation] = []
        for ann in img_id_to_anns.get(img_id, []):
            bbox = ann["bbox"]
            raw_name = cat_id_to_name.get(ann["category_id"], "unknown")
            merged_name = (
                class_merge_map.get(raw_name, raw_name) if class_merge_map else raw_name
            )
            if merged_name not in class_to_id:
                continue
            annotations.append(
                Annotation(
                    x1=bbox[0],
                    y1=bbox[1],
                    x2=bbox[0] + bbox[2],
                    y2=bbox[1] + bbox[3],
                    class_name=merged_name,
                    class_id=class_to_id[merged_name],
                )
            )

        slices = slice_image_with_annotations(
            image, annotations, slice_size, overlap, min_area_ratio
        )

        defect_slices = [s for s in slices if s.has_defect]
        negative_slices = [s for s in slices if not s.has_defect]

        if defect_slices:
            num_neg = int(
                len(defect_slices)
                * negative_sample_ratio
                / max(1, 1 - negative_sample_ratio)
            )
            num_neg = min(num_neg, len(negative_slices))
            sampled_neg = random.sample(negative_slices, num_neg) if num_neg > 0 else []
        else:
            num_neg = min(2, len(negative_slices))
            sampled_neg = random.sample(negative_slices, num_neg) if num_neg > 0 else []

        for s in defect_slices + sampled_neg:
            slice_filename = f"slice_{img_id_counter:06d}.jpg"
            slice_path = split_img_dir / slice_filename
            cv2.imwrite(str(slice_path), s.image)

            new_images.append(
                {
                    "id": img_id_counter,
                    "file_name": slice_filename,
                    "width": slice_size,
                    "height": slice_size,
                }
            )

            for sann in s.annotations:
                w = sann.x2 - sann.x1
                h = sann.y2 - sann.y1
                if w > 0 and h > 0:
                    new_annotations.append(
                        {
                            "id": ann_id_counter,
                            "image_id": img_id_counter,
                            "category_id": sann.class_id + 1,
                            "bbox": [sann.x1, sann.y1, w, h],
                            "area": w * h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id_counter += 1

            img_id_counter += 1
            if s.has_defect:
                total_defect_slices += 1
            else:
                total_negative_slices += 1

    new_categories = [
        {"id": idx + 1, "name": name} for idx, name in enumerate(target_classes)
    ]
    coco_output = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories,
    }

    ann_path = output_dir / "annotations" / f"instances_{split}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"切片完成({split}): {total_defect_slices} 个含缺陷切片, "
        f"{total_negative_slices} 个负样本切片, "
        f"共 {total_defect_slices + total_negative_slices} 个"
    )

    return str(output_dir)
