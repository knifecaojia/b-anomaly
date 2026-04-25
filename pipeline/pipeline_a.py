from __future__ import annotations

from core.config import SlicerConfig, TrainingConfig
from core.yolo_engine import PipelineA, YOLOEngine
from pipeline import BasePipeline, ImagePrediction


def create_pipeline_a(
    slicer_config: SlicerConfig | None = None,
    nms_iou: float = 0.5,
    batch_size: int = 8,
) -> PipelineA:
    return PipelineA(
        slicer_config=slicer_config,
        nms_iou=nms_iou,
        inference_batch_size=batch_size,
    )
