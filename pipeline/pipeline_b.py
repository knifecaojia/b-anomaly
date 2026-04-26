from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from loguru import logger

from core.anomalib_engine import AnomalibEngine
from core.anomaly_region_extractor import AnomalyRegionExtractor
from core.config import (
    AnomalibConfig,
    ClassifierConfig,
    PipelineBConfig,
    RegionExtractorConfig,
    SlicerConfig,
)
from core.defect_classifier import YOLODefectClassifier
from core.slicer import SliceCoord, slice_image_inference
from core.timing import TimingTracker
from pipeline import BasePipeline, BoundingBox, DetectionResult, ImagePrediction


class PipelineB(BasePipeline):
    def __init__(
        self,
        anomalib_config: Optional[AnomalibConfig] = None,
        region_extractor_config: Optional[RegionExtractorConfig] = None,
        classifier_config: Optional[ClassifierConfig] = None,
        slicer_config: Optional[SlicerConfig] = None,
    ) -> None:
        self._anomalib_config = anomalib_config or AnomalibConfig()
        self._re_config = region_extractor_config or RegionExtractorConfig()
        self._cls_config = classifier_config or ClassifierConfig()
        self._slicer_config = slicer_config or SlicerConfig()

        self._engine = AnomalibEngine()
        self._extractor = AnomalyRegionExtractor(
            threshold=self._re_config.threshold,
            min_area_ratio=self._re_config.min_area_ratio,
            morph_kernel_size=self._re_config.morph_kernel_size,
            max_regions=self._re_config.max_regions,
        )
        self._classifier: Optional[YOLODefectClassifier] = None
        self._ckpt_path: Optional[str] = None

    @property
    def class_names(self) -> List[str]:
        if self._classifier and self._classifier.is_loaded:
            return self._classifier.class_names
        return ["anomaly"]

    @property
    def is_loaded(self) -> bool:
        return self._engine.is_loaded

    def load_model(self, model_path: str) -> None:
        model_p = Path(model_path)
        if model_p.is_dir():
            ckpt_files = list(model_p.glob("*.ckpt")) + list(model_p.glob("*.pt"))
            if not ckpt_files:
                raise FileNotFoundError(f"目录中未找到模型文件: {model_path}")
            ckpt_path = str(ckpt_files[0])
        else:
            ckpt_path = model_path

        self._engine.load_model(ckpt_path)
        self._ckpt_path = ckpt_path

        if self._cls_config.model_path:
            self._classifier = YOLODefectClassifier(
                model_path=self._cls_config.model_path,
                conf_threshold=self._cls_config.conf_threshold,
                expand_ratio=self._cls_config.expand_ratio,
            )
            logger.info(f"Pipeline B 就绪 | 异常检测模型: {ckpt_path} | 分类器: {self._cls_config.model_path}")
        else:
            logger.info(f"Pipeline B 就绪 | 异常检测模型: {ckpt_path} | 无分类器（仅异常检测）")

    def train(self, config) -> None:
        if isinstance(config, PipelineBConfig):
            self._train_with_config(config)
        else:
            raise ValueError("Pipeline B 需要 PipelineBConfig 配置")

    def _train_with_config(self, config: PipelineBConfig) -> None:
        from dataset.prepare_anomalib_data import prepare_dataset

        data_dir = getattr(config, "_data_dir", "dataset/anomalib_data")
        output_dir = getattr(config, "_output_dir", "runs/anomalib")

        ac = config.anomalib
        ckpt_path = self._engine.train(
            data_dir=data_dir,
            output_dir=output_dir,
            backbone=ac.backbone,
            layers=ac.layers,
            coreset_sampling_ratio=ac.coreset_sampling_ratio,
            num_neighbors=ac.num_neighbors,
            image_size=ac.image_size,
            train_batch_size=ac.train_batch_size,
            eval_batch_size=ac.eval_batch_size,
            num_workers=ac.num_workers,
        )
        logger.info(f"PatchCore 训练完成，模型: {ckpt_path}")

    def predict(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> ImagePrediction:
        tracker = TimingTracker()
        tracker.start_step("加载图片")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")
        img_h, img_w = image.shape[:2]

        tracker.start_step("切片")
        full_anomaly_map: Optional[np.ndarray] = None

        if self._slicer_config.enabled and min(img_w, img_h) > 2 * self._slicer_config.slice_size:
            slices = slice_image_inference(
                image,
                slice_size=self._slicer_config.slice_size,
                overlap=self._slicer_config.overlap,
            )
            tracker.start_step("异常检测")

            tile_imgs = [s[0] for s in slices]
            tile_coords = [(c.x_start, c.y_start, c.x_start + c.width, c.y_start + c.height) for s, (_, c) in zip(slices, slices)]

            tile_results = self._engine.predict_tiles(
                tile_imgs,
                original_size=(img_h, img_w),
                tile_coords=tile_coords,
            )

            full_anomaly_map = self._engine.assemble_anomaly_map(
                tile_results,
                original_size=(img_h, img_w),
                tile_size=self._slicer_config.slice_size,
                overlap=self._slicer_config.overlap,
            )
        else:
            tracker.start_step("异常检测")
            anomaly_map, _ = self._engine.predict_single(image)
            full_anomaly_map = cv2.resize(
                anomaly_map, (img_w, img_h), interpolation=cv2.INTER_LINEAR
            )

        tracker.start_step("区域提取")
        regions = self._extractor.extract(full_anomaly_map, (img_h, img_w))
        logger.info(f"检测到 {len(regions)} 个异常区域")

        tracker.start_step("分类")
        detections: List[DetectionResult] = []

        if regions and self._classifier and self._classifier.is_loaded:
            region_tuples = [(r.x1, r.y1, r.x2, r.y2) for r in regions]
            classified = self._classifier.classify_regions(image, region_tuples)

            if classified:
                detections.extend(classified)
            else:
                for region in regions:
                    detections.append(
                        DetectionResult(
                            class_name="anomaly",
                            confidence=min(region.score, 1.0),
                            bbox=BoundingBox(
                                x1=region.x1, y1=region.y1,
                                x2=region.x2, y2=region.y2,
                            ),
                        )
                    )
        elif regions:
            for region in regions:
                detections.append(
                    DetectionResult(
                        class_name="anomaly",
                        confidence=min(region.score, 1.0),
                        bbox=BoundingBox(
                            x1=region.x1, y1=region.y1,
                            x2=region.x2, y2=region.y2,
                        ),
                    )
                )

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
