from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class SlicerConfig(BaseModel):
    enabled: bool = True
    slice_size: int = Field(default=640, ge=256, le=2048)
    overlap: float = Field(default=0.25, ge=0.0, le=0.5)
    min_area_ratio: float = Field(default=0.3, ge=0.1, le=1.0)
    negative_sample_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class AugmentationConfig(BaseModel):
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.3, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)


class TrainingConfig(BaseModel):
    model: str = "yolov8n.pt"
    dataset: str = ""
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)
    img_size: int = Field(default=640, ge=320)
    lr0: float = Field(default=0.01, gt=0)
    lrf: float = Field(default=0.01, ge=0)
    patience: int = Field(default=50, ge=1)
    device: str = "auto"
    output_dir: str = "runs"
    augmentation: AugmentationConfig = AugmentationConfig()
    slicer: SlicerConfig = SlicerConfig()


class APILogConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    retention_days: int = Field(default=30, ge=1)


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    model_path: str = ""
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    device: str = "auto"
    slicer: SlicerConfig = SlicerConfig()
    logging: APILogConfig = APILogConfig()
    api: APIConfig = APIConfig()


class ClassMapping(BaseModel):
    name: str
    merge_from: List[str]

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("类名不能为空")
        return v.strip()

    @field_validator("merge_from")
    @classmethod
    def merge_from_not_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("merge_from 不能为空列表")
        return v


class DefectMappingConfig(BaseModel):
    active_classes: List[ClassMapping]
    pending_classes: List[ClassMapping] = []

    def get_class_names(self) -> List[str]:
        return [c.name for c in self.active_classes]

    def get_merge_map(self) -> dict[str, str]:
        merge_map: dict[str, str] = {}
        for cls in self.active_classes:
            for src in cls.merge_from:
                merge_map[src] = cls.name
        return merge_map


class PipelineAConfig(BaseModel):
    training: TrainingConfig = TrainingConfig()


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_app_config(path: str | Path = "config/default.yaml") -> AppConfig:
    data = load_yaml(path)
    return AppConfig(**data)


def load_training_config(
    pipeline_path: str | Path = "config/pipeline_a.yaml",
    overrides: Optional[dict] = None,
) -> TrainingConfig:
    data = load_yaml(pipeline_path)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    return TrainingConfig(**data)


def load_defect_mapping(
    path: str | Path = "config/defect_mapping.yaml",
) -> DefectMappingConfig:
    data = load_yaml(path)
    return DefectMappingConfig(**data)


class AnomalibConfig(BaseModel):
    backbone: str = "wide_resnet50_2"
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = Field(default=0.1, ge=0.01, le=1.0)
    num_neighbors: int = Field(default=9, ge=1)
    image_size: int = Field(default=256, ge=64)
    anomaly_threshold: Optional[float] = None
    train_batch_size: int = Field(default=32, ge=1)
    eval_batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=0, ge=0)
    aug_enabled: bool = True
    aug_rotation_degrees: int = Field(default=25, ge=0, le=90)
    aug_crop_scale_min: float = Field(default=0.85, ge=0.5, le=1.0)
    aug_hflip_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    aug_vflip_prob: float = Field(default=0.5, ge=0.0, le=1.0)


class RegionExtractorConfig(BaseModel):
    threshold: Optional[float] = None
    min_area_ratio: float = Field(default=0.005, ge=0.0, le=1.0)
    morph_kernel_size: int = Field(default=3, ge=1)
    max_regions: int = Field(default=20, ge=1)
    expand_ratio: float = Field(default=0.1, ge=0.0, le=1.0)


class ClassifierConfig(BaseModel):
    type: str = "yolo"
    model_path: str = ""
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    expand_ratio: float = Field(default=0.2, ge=0.0, le=1.0)


class PipelineBConfig(BaseModel):
    anomalib: AnomalibConfig = AnomalibConfig()
    region_extractor: RegionExtractorConfig = RegionExtractorConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    slicer: SlicerConfig = SlicerConfig()


def load_pipeline_b_config(
    path: str | Path = "config/pipeline_b.yaml",
    overrides: Optional[dict] = None,
) -> PipelineBConfig:
    data = load_yaml(path)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    return PipelineBConfig(**data)


# ---------------------------------------------------------------------------
# Pipeline C — RF-DETR 目标检测管道配置
# ---------------------------------------------------------------------------

class RFDETRAugmentationConfig(BaseModel):
    """RF-DETR 数据增强配置"""
    enabled: bool = True
    preset: str = "AUG_INDUSTRIAL"
    custom: dict = {}


class RFDETRNMSConfig(BaseModel):
    """RF-DETR NMS 后处理配置"""
    enabled: bool = True
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class RFDETRTrainConfig(BaseModel):
    """RF-DETR 训练超参数配置"""
    model_variant: str = Field(default="s", pattern="^(n|s|m|l)$")
    pretrain_weights: str = ""
    resolution: int = Field(default=512, ge=128)
    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=4, ge=1)
    lr: float = Field(default=1e-4, gt=0)
    lr_encoder: float = Field(default=1e-5, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    use_ema: bool = True
    early_stopping: bool = False
    early_stopping_patience: int = Field(default=15, ge=1)
    gradient_checkpointing: bool = False
    dataset: str = ""
    device: str = "auto"
    output_dir: str = "runs"


class PipelineCConfig(BaseModel):
    """Pipeline C 完整配置"""
    training: RFDETRTrainConfig = RFDETRTrainConfig()
    slicer: SlicerConfig = SlicerConfig()
    nms: RFDETRNMSConfig = RFDETRNMSConfig()
    augmentation: RFDETRAugmentationConfig = RFDETRAugmentationConfig()


def load_pipeline_c_config(
    path: str | Path = "config/pipeline_c.yaml",
    overrides: Optional[dict] = None,
) -> PipelineCConfig:
    data = load_yaml(path)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    return PipelineCConfig(**data)
