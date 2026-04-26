import gc
import json
import sys
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

sys.path.insert(0, str(Path(__file__).parent))
from core.slicer import slice_image_inference

BASE = Path(__file__).parent
SLICE_DATA = BASE / "dataset" / "anomalib_P03_yiwu_sliced"
SOURCE_DATA = BASE / "dataset" / "anomalib_P03_yiwu"
RUNS_DIR = BASE / "runs"

SLICE_SIZE = 640
OVERLAP = 0.25
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def train_sliced_model():
    from core.anomalib_engine import AnomalibEngine

    output_dir = RUNS_DIR / "anomalib_P03_sliced"
    logger.info("训练切片版 PatchCore 模型")
    logger.info(f"训练数据: {SLICE_DATA}")
    logger.info(f"输出: {output_dir}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = AnomalibEngine()
    t0 = time.time()
    ckpt_path = engine.train(
        data_dir=str(SLICE_DATA),
        output_dir=str(output_dir),
        backbone="wide_resnet50_2",
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        image_size=256,
        train_batch_size=8,
        eval_batch_size=4,
        num_workers=0,
        aug_enabled=False,
    )
    elapsed = time.time() - t0
    logger.info(f"训练完成, 耗时 {elapsed:.1f}s, 模型: {ckpt_path}")
    return ckpt_path


def predict_image_with_slices(engine, image: np.ndarray) -> tuple[float, np.ndarray]:
    img_h, img_w = image.shape[:2]

    if min(img_w, img_h) > 2 * SLICE_SIZE:
        slices = slice_image_inference(image, slice_size=SLICE_SIZE, overlap=OVERLAP)
        tile_imgs = [s[0] for s in slices]
        tile_coords = [
            (c.x_start, c.y_start, c.x_start + c.width, c.y_start + c.height)
            for _, c in slices
        ]
        tile_results = engine.predict_tiles(
            tile_imgs, original_size=(img_h, img_w), tile_coords=tile_coords
        )

        max_score = max(score for _, score, _ in tile_results)

        full_map = engine.assemble_anomaly_map(
            tile_results, original_size=(img_h, img_w),
            tile_size=SLICE_SIZE, overlap=OVERLAP,
        )
        return max_score, full_map
    else:
        anomaly_map, score = engine.predict_single(image)
        return score, anomaly_map


def evaluate_sliced_model(ckpt_path: str):
    from core.anomalib_engine import AnomalibEngine

    logger.info("评估切片版模型")
    logger.info("推理方式: 对测试大图切片推理, 取所有切片最大分数作为图片分数")

    engine = AnomalibEngine()
    engine.load_model(ckpt_path)

    test_normal_dir = SOURCE_DATA / "test" / "normal"
    test_abnormal_dir = SOURCE_DATA / "test" / "abnormal"

    normal_scores = _evaluate_directory(engine, test_normal_dir, "test_normal")
    abnormal_scores = _evaluate_directory(engine, test_abnormal_dir, "test_abnormal")

    normal_arr = np.array(normal_scores)
    abnormal_arr = np.array(abnormal_scores)

    if len(normal_arr) == 0 or len(abnormal_arr) == 0:
        logger.warning("评分数据不足")
        return {}

    all_scores = np.concatenate([normal_arr, abnormal_arr])
    all_labels = np.concatenate([np.zeros(len(normal_arr)), np.ones(len(abnormal_arr))])

    best_f1, best_thresh, best_precision, best_recall = _find_best_threshold(
        all_scores, all_labels
    )

    sep = float(abnormal_arr.mean() - normal_arr.mean())

    result = {
        "method": "切片训练+切片推理",
        "slice_size": SLICE_SIZE,
        "overlap": OVERLAP,
        "train_slices": 22176,
        "normal_count": int(len(normal_arr)),
        "abnormal_count": int(len(abnormal_arr)),
        "normal_mean": float(normal_arr.mean()),
        "normal_std": float(normal_arr.std()),
        "normal_min": float(normal_arr.min()),
        "normal_max": float(normal_arr.max()),
        "normal_median": float(np.median(normal_arr)),
        "abnormal_mean": float(abnormal_arr.mean()),
        "abnormal_std": float(abnormal_arr.std()),
        "abnormal_min": float(abnormal_arr.min()),
        "abnormal_max": float(abnormal_arr.max()),
        "abnormal_median": float(np.median(abnormal_arr)),
        "separation": sep,
        "best_threshold": float(best_thresh),
        "best_f1": float(best_f1),
        "best_precision": float(best_precision),
        "best_recall": float(best_recall),
    }

    logger.info("=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(
        f"  正常品: {result['normal_count']} 张, "
        f"均值={result['normal_mean']:.2f}, "
        f"中位数={result['normal_median']:.2f}, "
        f"范围=[{result['normal_min']:.2f}, {result['normal_max']:.2f}]"
    )
    logger.info(
        f"  缺陷品: {result['abnormal_count']} 张, "
        f"均值={result['abnormal_mean']:.2f}, "
        f"中位数={result['abnormal_median']:.2f}, "
        f"范围=[{result['abnormal_min']:.2f}, {result['abnormal_max']:.2f}]"
    )
    logger.info(f"  分离度: {result['separation']:.2f}")
    logger.info(f"  最佳阈值: {result['best_threshold']:.2f}")
    logger.info(
        f"  F1={result['best_f1']:.4f}, "
        f"精确率={result['best_precision']:.4f}, "
        f"召回率={result['best_recall']:.4f}"
    )

    return result


def _evaluate_directory(engine, img_dir: Path, label: str) -> list[float]:
    images = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS])
    scores = []
    for i, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        score, _ = predict_image_with_slices(engine, img)
        scores.append(score)
        del img
        gc.collect()
        if (i + 1) % 10 == 0:
            logger.info(f"  [{label}] 已推理 {i+1}/{len(images)}")
    logger.info(f"  [{label}] 完成: {len(scores)} 张")
    return scores


def _find_best_threshold(all_scores, all_labels):
    thresholds = np.percentile(all_scores, np.arange(1, 100, 0.5))
    best_f1 = 0
    best_thresh = 0
    best_precision = 0
    best_recall = 0
    for t in thresholds:
        preds = (all_scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (all_labels == 1)))
        fp = int(np.sum((preds == 1) & (all_labels == 0)))
        fn = int(np.sum((preds == 0) & (all_labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_precision = precision
            best_recall = recall
    return best_f1, best_thresh, best_precision, best_recall


def main():
    logger.info("=" * 60)
    logger.info("切片训练法（方案A）")
    logger.info("=" * 60)

    ckpt_path = train_sliced_model()

    if not ckpt_path:
        logger.error("训练失败")
        return

    result = evaluate_sliced_model(ckpt_path)

    if result:
        summary_path = RUNS_DIR / "sliced_comparison_results.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"\n结果已保存至: {summary_path}")


if __name__ == "__main__":
    main()
