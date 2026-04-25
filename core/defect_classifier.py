from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from core.yolo_engine import YOLOEngine
from pipeline import BoundingBox, DetectionResult


class YOLODefectClassifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        expand_ratio: float = 0.2,
    ) -> None:
        self._engine = YOLOEngine()
        self._conf_threshold = conf_threshold
        self._expand_ratio = expand_ratio
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        self._engine.load_model(model_path)
        logger.info(f"分类器模型已加载: {model_path}")

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    @property
    def class_names(self) -> List[str]:
        return self._engine.class_names

    def classify_region(
        self,
        image: np.ndarray,
        region_bbox: Tuple[float, float, float, float],
    ) -> List[DetectionResult]:
        if not self._engine.is_loaded:
            return []

        img_h, img_w = image.shape[:2]
        x1, y1, x2, y2 = region_bbox

        bw = x2 - x1
        bh = y2 - y1
        expand_w = bw * self._expand_ratio
        expand_h = bh * self._expand_ratio

        crop_x1 = max(0, int(x1 - expand_w))
        crop_y1 = max(0, int(y1 - expand_h))
        crop_x2 = min(img_w, int(x2 + expand_w))
        crop_y2 = min(img_h, int(y2 + expand_h))

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return []

        detections = self._engine.predict_single(crop, self._conf_threshold)

        mapped = []
        for det in detections:
            mapped.append(
                DetectionResult(
                    class_name=det.class_name,
                    confidence=det.confidence,
                    bbox=BoundingBox(
                        x1=det.bbox.x1 + crop_x1,
                        y1=det.bbox.y1 + crop_y1,
                        x2=det.bbox.x2 + crop_x1,
                        y2=det.bbox.y2 + crop_y1,
                    ),
                )
            )

        return mapped

    def classify_regions(
        self,
        image: np.ndarray,
        regions: List[Tuple[float, float, float, float]],
    ) -> List[DetectionResult]:
        all_detections: List[DetectionResult] = []
        seen_boxes: List[Tuple[float, float, float, float]] = []

        for region in regions:
            dets = self.classify_region(image, region)
            for det in dets:
                is_dup = False
                for sb in seen_boxes:
                    if self._iou(
                        (det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2), sb
                    ) > 0.5:
                        is_dup = True
                        break
                if not is_dup:
                    all_detections.append(det)
                    seen_boxes.append((det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2))

        return all_detections

    @staticmethod
    def _iou(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
