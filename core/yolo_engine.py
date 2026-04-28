from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from core.config import SlicerConfig, TrainingConfig
from core.device import get_device
from core.slicer import (
    SliceCoord,
    map_detections_to_global,
    nms_across_slices,
    slice_image_inference,
)
from core.timing import TimingTracker
from pipeline import BasePipeline, BoundingBox, DetectionResult, ImagePrediction


class YOLOEngine:
    def __init__(self) -> None:
        self._model = None
        self._model_path: Optional[str] = None
        self._class_names: List[str] = []

    def load_model(self, model_path: str) -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._model_path = model_path
        self._class_names = list(self._model.names.values())
        logger.info(f"模型已加载: {model_path}, 类别: {self._class_names}")

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def predict_single(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> List[DetectionResult]:
        if self._model is None:
            raise RuntimeError("模型未加载")

        results = self._model(image, conf=conf_threshold, verbose=False)
        detections: List[DetectionResult] = []

        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    cls_name = self._class_names[cls_ids[i]] if cls_ids[i] < len(self._class_names) else f"class_{cls_ids[i]}"
                    detections.append(
                        DetectionResult(
                            class_name=cls_name,
                            confidence=float(confs[i]),
                            bbox=BoundingBox(
                                x1=float(boxes[i][0]),
                                y1=float(boxes[i][1]),
                                x2=float(boxes[i][2]),
                                y2=float(boxes[i][3]),
                            ),
                        )
                    )

        return detections

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_threshold: float = 0.25,
    ) -> List[List[DetectionResult]]:
        if self._model is None:
            raise RuntimeError("模型未加载")

        results = self._model(images, conf=conf_threshold, verbose=False)
        batch_detections: List[List[DetectionResult]] = []

        for r in results:
            detections: List[DetectionResult] = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)

                for i in range(len(boxes)):
                    cls_name = self._class_names[cls_ids[i]] if cls_ids[i] < len(self._class_names) else f"class_{cls_ids[i]}"
                    detections.append(
                        DetectionResult(
                            class_name=cls_name,
                            confidence=float(confs[i]),
                            bbox=BoundingBox(
                                x1=float(boxes[i][0]),
                                y1=float(boxes[i][1]),
                                x2=float(boxes[i][2]),
                                y2=float(boxes[i][3]),
                            ),
                        )
                    )
            batch_detections.append(detections)

        return batch_detections

    def train(
        self,
        config: TrainingConfig,
        dataset_yaml: str,
    ) -> str:
        from ultralytics import YOLO

        device = get_device() if config.device == "auto" else config.device
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = config.output_dir
        run_name = f"train_{timestamp}"

        model = YOLO(config.model)

        results = model.train(
            data=dataset_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.img_size,
            lr0=config.lr0,
            lrf=config.lrf,
            patience=config.patience,
            device=device,
            project=project_name,
            name=run_name,
            mosaic=config.augmentation.mosaic,
            copy_paste=config.augmentation.copy_paste,
            fliplr=config.augmentation.fliplr,
            flipud=config.augmentation.flipud,
            hsv_h=config.augmentation.hsv_h,
            hsv_s=config.augmentation.hsv_s,
            hsv_v=config.augmentation.hsv_v,
            workers=0,
            exist_ok=True,
        )

        run_dir = Path(results.save_dir)
        self._save_metadata(run_dir, config, dataset_yaml)

        best_path = str(run_dir / "weights" / "best.pt")
        logger.info(f"训练完成，最佳模型: {best_path}")
        return best_path

    def _save_metadata(
        self, run_dir: Path, config: TrainingConfig, dataset_yaml: str
    ) -> None:
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": config.model,
            "dataset_yaml": dataset_yaml,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "img_size": config.img_size,
            "device": config.device,
            "augmentation": config.augmentation.model_dump(),
            "slicer": config.slicer.model_dump(),
        }

        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            with open(results_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    for key in [
                        "metrics/mAP50(B)",
                        "metrics/mAP50-95(B)",
                        "metrics/precision(B)",
                        "metrics/recall(B)",
                    ]:
                        if key in last:
                            short_key = key.replace("metrics/", "").replace("(B)", "")
                            metadata[f"final_{short_key}"] = last[key]

        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)


class PipelineA(BasePipeline):
    def __init__(
        self,
        slicer_config: Optional[SlicerConfig] = None,
        nms_iou: float = 0.5,
        inference_batch_size: int = 8,
    ) -> None:
        self._engine = YOLOEngine()
        self._slicer_config = slicer_config or SlicerConfig()
        self._nms_iou = nms_iou
        self._inference_batch_size = inference_batch_size

    def load_model(self, model_path: str) -> None:
        self._engine.load_model(model_path)

    @property
    def class_names(self) -> List[str]:
        return self._engine.class_names

    def train(self, config: TrainingConfig) -> None:
        from core.dataset_manager import DatasetManager
        from core.slicer import generate_training_slices

        dm = DatasetManager()
        coco_dir = Path(config.dataset)

        class_merge_map = None
        merge_map_path = Path("config/defect_mapping.yaml")
        if merge_map_path.exists():
            from core.config import load_defect_mapping
            dm_config = load_defect_mapping()
            class_merge_map = dm_config.get_merge_map()

        if config.slicer.enabled:
            slicer_out = coco_dir.parent / f"{coco_dir.name}_sliced"
            need_slicing = not (
                slicer_out / "annotations" / "instances_train.json"
            ).exists()

            if need_slicing:
                logger.info("开始切片预处理...")
                for split in ["train", "val", "test"]:
                    ann_path = dm._find_ann_path(coco_dir, split)
                    img_dir = dm._find_img_dir(coco_dir, split)
                    if not ann_path or not img_dir:
                        continue
                    logger.info(f"切片 {split}: 标注={ann_path}, 图片={img_dir}")
                    generate_training_slices(
                        image_dir=img_dir,
                        coco_annotation_path=str(ann_path),
                        output_dir=slicer_out,
                        slice_size=config.slicer.slice_size,
                        overlap=config.slicer.overlap,
                        min_area_ratio=config.slicer.min_area_ratio,
                        negative_sample_ratio=config.slicer.negative_sample_ratio,
                        class_merge_map=class_merge_map,
                        split=split,
                    )
            else:
                logger.info("切片数据已存在，跳过切片步骤")

            yolo_dir = coco_dir.parent / f"{coco_dir.name}_yolo"
            dataset_yaml = dm.prepare_yolo_dataset(slicer_out, yolo_dir)
        else:
            logger.info("切片已禁用，直接使用原始数据集")
            yolo_dir = coco_dir.parent / f"{coco_dir.name}_yolo_direct"
            dataset_yaml = dm.prepare_yolo_dataset(coco_dir, yolo_dir)

        best_model = self._engine.train(config, dataset_yaml)
        logger.info(f"训练完成，最佳模型: {best_model}")

    def predict(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> ImagePrediction:
        tracker = TimingTracker()
        tracker.start_step("加载图片")

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_h, img_w = image.shape[:2]
        tracker.start_step("切片")

        all_boxes: List[np.ndarray] = []
        all_scores: List[float] = []
        all_class_ids: List[int] = []
        all_class_names: List[str] = []

        if self._slicer_config.enabled and min(img_w, img_h) > 2 * self._slicer_config.slice_size:
            slices = slice_image_inference(
                image,
                slice_size=self._slicer_config.slice_size,
                overlap=self._slicer_config.overlap,
            )
            tracker.start_step("推理")

            batch_imgs = [s[0] for s in slices]
            batch_coords = [s[1] for s in slices]

            for i in range(0, len(batch_imgs), self._inference_batch_size):
                batch = batch_imgs[i : i + self._inference_batch_size]
                coords = batch_coords[i : i + self._inference_batch_size]
                batch_results = self._engine.predict_batch(batch, conf_threshold)

                for j, detections in enumerate(batch_results):
                    for det in detections:
                        local_box = np.array(
                            [det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2]
                        ).reshape(1, 4)
                        global_box = map_detections_to_global(local_box, coords[j])
                        all_boxes.append(global_box[0])
                        all_scores.append(det.confidence)
                        cls_id = 0
                        for ci, cn in enumerate(self._engine.class_names):
                            if cn == det.class_name:
                                cls_id = ci
                                break
                        all_class_ids.append(cls_id)
                        all_class_names.append(det.class_name)

            tracker.start_step("NMS合并")
            if all_boxes:
                boxes_arr = np.array(all_boxes)
                scores_arr = np.array(all_scores)
                class_arr = np.array(all_class_ids)

                kept_boxes, kept_scores, kept_classes = nms_across_slices(
                    boxes_arr, scores_arr, class_arr, self._nms_iou
                )

                detections = []
                for i in range(len(kept_boxes)):
                    detections.append(
                        DetectionResult(
                            class_name=self._engine.class_names[kept_classes[i]],
                            confidence=float(kept_scores[i]),
                            bbox=BoundingBox(
                                x1=float(kept_boxes[i][0]),
                                y1=float(kept_boxes[i][1]),
                                x2=float(kept_boxes[i][2]),
                                y2=float(kept_boxes[i][3]),
                            ),
                        )
                    )
            else:
                detections = []
        else:
            tracker.start_step("推理")
            detections = self._engine.predict_single(image, conf_threshold)

        tracker.start_step("后处理")
        normalized_dets = [d.to_normalized(img_w, img_h) for d in detections]
        tracker.finish_all()

        return ImagePrediction(
            image_path=image_path,
            detections=normalized_dets,
            timing_ms=tracker.get_results(),
        )

    def predict_batch(
        self, image_paths: List[str], conf_threshold: float = 0.25
    ) -> List[ImagePrediction]:
        return [self.predict(p, conf_threshold) for p in image_paths]
