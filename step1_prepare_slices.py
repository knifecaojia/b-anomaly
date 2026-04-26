import gc
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

sys.path.insert(0, str(Path(__file__).parent))
from core.slicer import compute_slice_coords, slice_image_inference

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset"
SOURCE_DATA = DATASET_DIR / "anomalib_P03_yiwu"
SLICE_DATA = DATASET_DIR / "anomalib_P03_yiwu_sliced"
RUNS_DIR = BASE / "runs"

SLICE_SIZE = 640
OVERLAP = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def prepare_sliced_dataset():
    logger.info("准备切片训练数据集")
    logger.info(f"源数据: {SOURCE_DATA}")
    logger.info(f"输出: {SLICE_DATA}")
    logger.info(f"切片大小: {SLICE_SIZE}, 重叠率: {OVERLAP}")

    if SLICE_DATA.exists():
        shutil.rmtree(SLICE_DATA)

    train_normal_src = SOURCE_DATA / "train" / "normal"
    test_normal_src = SOURCE_DATA / "test" / "normal"
    test_abnormal_src = SOURCE_DATA / "test" / "abnormal"

    train_normal_dst = SLICE_DATA / "train" / "normal"
    test_normal_dst = SLICE_DATA / "test" / "normal"
    test_abnormal_dst = SLICE_DATA / "test" / "abnormal"

    stats = {}

    stats["train_normal"] = _slice_directory(
        train_normal_src, train_normal_dst, "train_normal"
    )

    stats["test_normal"] = _slice_directory(
        test_normal_src, test_normal_dst, "test_normal"
    )

    stats["test_abnormal"] = _slice_directory(
        test_abnormal_src, test_abnormal_dst, "test_abnormal"
    )

    summary = {
        "source": str(SOURCE_DATA),
        "output": str(SLICE_DATA),
        "slice_size": SLICE_SIZE,
        "overlap": OVERLAP,
        "stats": stats,
    }
    summary_path = SLICE_DATA / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("切片数据集准备完成")
    logger.info(f"  train/normal: {stats['train_normal']['slices']} 张切片 (来自 {stats['train_normal']['images']} 张原图)")
    logger.info(f"  test/normal: {stats['test_normal']['slices']} 张切片 (来自 {stats['test_normal']['images']} 张原图)")
    logger.info(f"  test/abnormal: {stats['test_abnormal']['slices']} 张切片 (来自 {stats['test_abnormal']['images']} 张原图)")
    logger.info("=" * 60)

    return SLICE_DATA


def _slice_directory(src_dir: Path, dst_dir: Path, label: str) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in src_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS])

    total_slices = 0
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        coords = compute_slice_coords(img_w, img_h, SLICE_SIZE, OVERLAP)

        for j, coord in enumerate(coords):
            crop = img[
                coord.y_start : coord.y_start + coord.height,
                coord.x_start : coord.x_start + coord.width,
            ]

            if coord.width < SLICE_SIZE or coord.height < SLICE_SIZE:
                padded = np.zeros((SLICE_SIZE, SLICE_SIZE, 3), dtype=img.dtype)
                padded[: crop.shape[0], : crop.shape[1]] = crop
                crop = padded

            slice_name = f"{img_path.stem}_s{j:03d}.jpg"
            cv2.imwrite(str(dst_dir / slice_name), crop)
            total_slices += 1

        del img
        gc.collect()

        if (i + 1) % 20 == 0:
            logger.info(f"  [{label}] 已处理 {i+1}/{len(images)} 张原图, 累计 {total_slices} 张切片")

    logger.info(f"  [{label}] 完成: {len(images)} 张原图 -> {total_slices} 张切片")
    return {"images": len(images), "slices": total_slices}


if __name__ == "__main__":
    prepare_sliced_dataset()
