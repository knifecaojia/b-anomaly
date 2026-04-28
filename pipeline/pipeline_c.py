import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from core.config import PipelineCConfig, load_pipeline_c_config
from core.rfdetr_engine import RFDETREngine
from pipeline import BasePipeline, ImagePrediction, DetectionResult, BoundingBox


class PipelineC(BasePipeline):
    """
    Pipeline C: 纯 RF-DETR 端到端目标检测
    支持超大图切片检测 (Slicing) 与 NMS 合并
    """

    def __init__(
        self,
        config: PipelineCConfig,
        class_names: Optional[list[str]] = None,
    ):
        self.config = config
        self.class_names = class_names or []
        self.engine = RFDETREngine(config=self.config, class_names=self.class_names)
        self._is_initialized = False
        self._slicer_config = self.config.slicer

    @property
    def is_loaded(self) -> bool:
        return self._is_initialized

    @property
    def _ckpt_path(self) -> str:
        return self.config.training.pretrain_weights

    def load_model(self, model_path: str) -> None:
        self.config.training.pretrain_weights = model_path
        self.initialize()

    def train(self, config) -> None:
        self.engine.train(dataset_dir=self.config.training.dataset)

    def initialize(self) -> None:
        """初始化加载模型与预训练权重"""
        if self._is_initialized:
            return
            
        logger.info("初始化 Pipeline C (RF-DETR)...")
        # 如果提供了预训练权重路径，则加载
        self.engine.build_model()
        self.class_names = self.engine.class_names
        self._is_initialized = True
        logger.info(f"Pipeline C 初始化完成. 类别: {self.class_names}")

    def process(
        self,
        image_path: str | Path,
        conf_threshold: float = 0.25,
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        处理单张图片
        返回:
            (结果字典, 渲染后的图像)
        """
        if not self._is_initialized:
            self.initialize()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"找不到图片文件: {image_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片文件: {image_path}")

        start_time = time.time()
        
        # 引擎已内部封装了 切片 -> 推理 -> 全局坐标映射 -> NMS 整合
        detections = self.engine.predict(
            image=image,
            conf_threshold=conf_threshold,
        )

        proc_time = time.time() - start_time

        # 生成结果数据结构
        results = {
            "image_path": str(image_path),
            "detections": detections,
            "processing_time_ms": proc_time * 1000,
            "has_defect": len(detections) > 0,
            "pipeline": "pipeline_c",
            "model_variant": self.config.training.model_variant,
        }

        # 可视化渲染
        vis_image = self.render(image, detections)

        return results, vis_image

    def predict(
        self, image_path: str, conf_threshold: float = 0.25
    ) -> ImagePrediction:
        results, _ = self.process(image_path, conf_threshold)
        
        det_results = []
        for d in results["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            det_results.append(
                DetectionResult(
                    class_name=d["class_name"],
                    confidence=d["confidence"],
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
            
        timing_ms = {"total": results["processing_time_ms"]}
        
        # 归一化 (PipelineA/B返回normalized的bbox)
        # 但是PipelineC的predict引擎返回的可能已经是原图坐标了。
        # 等等，如果需要normalized的bbox，需要在predict里转换。
        image = cv2.imread(str(image_path))
        img_h, img_w = image.shape[:2]
        normalized_dets = [d.to_normalized(img_w, img_h) for d in det_results]
        
        return ImagePrediction(
            image_path=str(image_path),
            detections=normalized_dets,
            timing_ms=timing_ms,
        )

    def predict_batch(
        self, image_paths: list[str], conf_threshold: float = 0.25
    ) -> list[ImagePrediction]:
        return [self.predict(p, conf_threshold) for p in image_paths]

    def render(self, image: np.ndarray, detections: list[Dict[str, Any]]) -> np.ndarray:
        """渲染检测框与置信度到原图"""
        vis_img = image.copy()
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            cname = det["class_name"]

            # 红色粗框
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # 标签文本
            label = f"{cname} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            # 背景色块
            cv2.rectangle(vis_img, (x1, y1 - th - baseline - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
            # 白色文字
            cv2.putText(vis_img, label, (x1 + 5, y1 - baseline - 5), font, font_scale, (255, 255, 255), thickness)

        return vis_img
