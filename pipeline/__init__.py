from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def normalize(self, img_w: int, img_h: int) -> "BoundingBox":
        return BoundingBox(
            x1=self.x1 / img_w,
            y1=self.y1 / img_h,
            x2=self.x2 / img_w,
            y2=self.y2 / img_h,
        )

    def to_region_str(self) -> str:
        return f"{self.x1:.6f},{self.y1:.6f},{self.x2:.6f},{self.y2:.6f}"


@dataclass
class DetectionResult:
    class_name: str
    confidence: float
    bbox: BoundingBox

    def to_normalized(self, img_w: int, img_h: int) -> "DetectionResult":
        return DetectionResult(
            class_name=self.class_name,
            confidence=self.confidence,
            bbox=self.bbox.normalize(img_w, img_h),
        )


@dataclass
class ImagePrediction:
    image_path: str
    detections: List[DetectionResult] = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)


class BasePipeline(ABC):
    @abstractmethod
    def train(self, config) -> None:
        ...

    @abstractmethod
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> ImagePrediction:
        ...

    @abstractmethod
    def predict_batch(self, image_paths: List[str], conf_threshold: float = 0.25) -> List[ImagePrediction]:
        ...

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        ...
