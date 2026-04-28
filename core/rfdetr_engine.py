import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from core.config import PipelineCConfig
from core.slicer import slice_image_inference, map_detections_to_global, nms_across_slices


class RFDETREngine:
    """RF-DETR 推理与训练引擎包装"""

    def __init__(self, config: PipelineCConfig, class_names: Optional[List[str]] = None):
        self.config = config
        self.class_names = class_names or []
        self.model = None

    def _get_model_class(self):
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

        variant_map = {
            "n": RFDETRNano,
            "s": RFDETRSmall,
            "m": RFDETRMedium,
            "l": RFDETRLarge,
        }
        variant = self.config.training.model_variant.lower()
        if variant not in variant_map:
            logger.warning(f"未知的 model_variant: {variant}，回退使用 Small 模型")
            return RFDETRSmall
        return variant_map[variant]

    def build_model(self, pretrain_weights: str = "", num_classes: Optional[int] = None):
        """构建模型实例，按需加载预训练权重"""
        ModelClass = self._get_model_class()
        kwargs = {}
        
        # 优先使用显式指定的分类数
        if num_classes is not None:
            kwargs["num_classes"] = num_classes
        elif self.class_names:
            kwargs["num_classes"] = len(self.class_names)

        # 优先使用传入的预训练权重路径，否则使用配置中的路径
        weights = pretrain_weights or self.config.training.pretrain_weights
        if weights:
            kwargs["pretrain_weights"] = str(weights)

        try:
            self.model = ModelClass(**kwargs)
            if self.model.class_names and not self.class_names:
                self.class_names = self.model.class_names
        except Exception as e:
            logger.error(f"加载 RF-DETR 模型失败: {e}")
            raise e

    def train(self, dataset_dir: str | Path) -> str:
        """执行 RF-DETR 训练流程"""
        if not self.model:
            self.build_model()

        tc = self.config.training
        output_dir = Path(tc.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        aug_config = None
        if self.config.augmentation.enabled:
            from rfdetr.datasets.aug_config import AUG_INDUSTRIAL
            # TODO: 支持从 self.config.augmentation.custom 解析
            aug_config = AUG_INDUSTRIAL

        logger.info(f"开始训练 RF-DETR ({tc.model_variant.upper()})")
        logger.info(f"数据集: {dataset_dir}")
        logger.info(f"输出目录: {output_dir}")

        self.model.train(
            dataset_dir=str(dataset_dir),
            epochs=tc.epochs,
            batch_size=tc.batch_size,
            grad_accum_steps=tc.grad_accum_steps,
            lr=tc.lr,
            lr_encoder=tc.lr_encoder,
            weight_decay=tc.weight_decay,
            output_dir=str(output_dir),
            aug_config=aug_config,
            early_stopping=tc.early_stopping,
            early_stopping_patience=tc.early_stopping_patience,
            use_ema=tc.use_ema,
            class_names=self.class_names if self.class_names else None,
        )

        ckpt_path = output_dir / "checkpoint_best_total.pth"
        if not ckpt_path.exists():
            ckpt_path = output_dir / "last.ckpt"

        logger.info(f"训练完成，模型保存在: {ckpt_path}")
        return str(ckpt_path)

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        slicer_config: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """执行单张图片推理，支持自动切片和 NMS 合并"""
        if not self.model:
            raise RuntimeError("模型未加载，请先调用 build_model(pretrain_weights=...)")

        slicer_cfg = slicer_config if slicer_config else self.config.slicer
        
        # 1. 如果不启用切片，或者图片很小，直接全图推理
        if not slicer_cfg.enabled:
            return self._predict_single(image, conf_threshold, offset_x=0, offset_y=0)

        # 2. 切片推理
        slices = slice_image_inference(
            image,
            slice_size=slicer_cfg.slice_size,
            overlap=slicer_cfg.overlap,
        )

        all_boxes = []
        all_scores = []
        all_class_ids = []

        for crop, coord in slices:
            # 局部推理
            detections = self.model.predict(crop, threshold=conf_threshold)
            
            if len(detections.confidence) == 0:
                continue
                
            local_boxes = detections.xyxy
            scores = detections.confidence
            class_ids = detections.class_id
            
            # 映射回全局坐标
            global_boxes = map_detections_to_global(local_boxes, coord)
            
            all_boxes.append(global_boxes)
            all_scores.append(scores)
            all_class_ids.append(class_ids)

        if not all_boxes:
            return []

        # 合并所有切片的结果
        merged_boxes = np.vstack(all_boxes)
        merged_scores = np.concatenate(all_scores)
        merged_class_ids = np.concatenate(all_class_ids)

        # 3. NMS 后处理去重
        if self.config.nms.enabled:
            merged_boxes, merged_scores, merged_class_ids = nms_across_slices(
                merged_boxes,
                merged_scores,
                merged_class_ids,
                iou_threshold=self.config.nms.iou_threshold,
            )

        results = []
        for i in range(len(merged_scores)):
            cid = int(merged_class_ids[i])
            cname = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            box = merged_boxes[i].tolist()
            results.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": float(merged_scores[i]),
                "bbox": box,
            })

        return results

    def _predict_single(
        self,
        image: np.ndarray,
        conf_threshold: float,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> List[Dict[str, Any]]:
        """单张/单个切片的基础推理函数"""
        detections = self.model.predict(image, threshold=conf_threshold)
        
        results = []
        if len(detections.confidence) == 0:
            return results

        boxes = detections.xyxy
        scores = detections.confidence
        class_ids = detections.class_id

        for i in range(len(scores)):
            cid = int(class_ids[i])
            cname = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            x1, y1, x2, y2 = boxes[i]
            results.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": float(scores[i]),
                "bbox": [
                    float(x1 + offset_x),
                    float(y1 + offset_y),
                    float(x2 + offset_x),
                    float(y2 + offset_y),
                ],
            })
            
        return results