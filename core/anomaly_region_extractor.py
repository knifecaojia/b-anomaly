from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class AnomalyRegion:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


class AnomalyRegionExtractor:
    def __init__(
        self,
        threshold: Optional[float] = None,
        min_area_ratio: float = 0.005,
        morph_kernel_size: int = 3,
        max_regions: int = 20,
    ) -> None:
        self._threshold = threshold
        self._min_area_ratio = min_area_ratio
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._max_regions = max_regions

    def extract(
        self,
        anomaly_map: np.ndarray,
        image_size: Tuple[int, int],
    ) -> List[AnomalyRegion]:
        if anomaly_map.max() <= 0:
            return []

        h, w = image_size[:2]
        resized_map = cv2.resize(anomaly_map, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = self._threshold_map(resized_map)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)

        min_area = int(h * w * self._min_area_ratio)
        regions = self._find_connected_regions(mask, resized_map, min_area)

        regions.sort(key=lambda r: r.score, reverse=True)
        return regions[: self._max_regions]

    def _threshold_map(self, anomaly_map: np.ndarray) -> np.ndarray:
        norm_map = anomaly_map.copy()
        if norm_map.max() > 1.0:
            norm_map = norm_map / norm_map.max()

        if self._threshold is not None:
            _, mask = cv2.threshold(
                (norm_map * 255).astype(np.uint8),
                int(self._threshold * 255),
                255,
                cv2.THRESH_BINARY,
            )
        else:
            _, mask = cv2.threshold(
                (norm_map * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

        return mask

    def _find_connected_regions(
        self,
        mask: np.ndarray,
        score_map: np.ndarray,
        min_area: int,
    ) -> List[AnomalyRegion]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        regions: List[AnomalyRegion] = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            component_mask = (labels == i).astype(np.uint8)
            region_score = float(np.max(score_map * component_mask))

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            regions.append(
                AnomalyRegion(
                    x1=float(x),
                    y1=float(y),
                    x2=float(x + bw),
                    y2=float(y + bh),
                    score=region_score,
                )
            )

        return regions
