from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class AnomalibEngine:
    def __init__(self) -> None:
        self._model = None
        self._engine = None
        self._inferencer = None
        self._ckpt_path: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._inferencer is not None or self._model is not None

    def train(
        self,
        data_dir: str,
        output_dir: str = "runs/anomalib",
        backbone: str = "wide_resnet50_2",
        layers: Optional[List[str]] = None,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        image_size: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        max_epochs: int = 1,
    ) -> str:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        if layers is None:
            layers = ["layer2", "layer3"]

        data_dir = Path(data_dir)
        datamodule = Folder(
            root=str(data_dir),
            normal_dir="train/normal",
            abnormal_dir="test/abnormal",
            normal_test_dir="test/normal",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            image_size=(image_size, image_size),
        )

        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )

        engine = Engine(
            default_root_dir=output_dir,
            max_epochs=max_epochs,
        )

        engine.fit(datamodule=datamodule, model=model)

        ckpt_path = Path(output_dir) / "patchcore.ckpt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        best_path = engine.best_model_path
        if best_path:
            shutil.copy2(best_path, str(ckpt_path))
            logger.info(f"模型已保存到: {ckpt_path}")
        else:
            logger.warning("未找到最佳模型路径，尝试使用引擎内部模型导出")
            try:
                export_path = Path(output_dir) / "patchcore.pt"
                engine.export(
                    model=model,
                    export_type="torch",
                    export_root=str(output_dir),
                )
                if export_path.exists():
                    logger.info(f"模型已导出到: {export_path}")
                    self._ckpt_path = str(export_path)
                    return self._ckpt_path
            except Exception:
                pass
            logger.error("模型保存失败")

        self._model = model
        self._engine = engine
        self._ckpt_path = str(ckpt_path)
        return self._ckpt_path

    def load_model(self, ckpt_path: str) -> None:
        from anomalib.deploy import TorchInferencer

        self._inferencer = TorchInferencer(path=ckpt_path)
        self._ckpt_path = ckpt_path
        logger.info(f"Anomalib 模型已加载: {ckpt_path}")

    def predict_single(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        if self._inferencer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        result = self._inferencer.predict(image=image)

        anomaly_map = result.anomaly_map
        if hasattr(anomaly_map, "cpu"):
            anomaly_map = anomaly_map.cpu().numpy()
        elif hasattr(anomaly_map, "numpy"):
            anomaly_map = anomaly_map.numpy()
        if isinstance(anomaly_map, np.ndarray) and anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.squeeze(0)

        pred_score = result.pred_score
        if hasattr(pred_score, "item"):
            pred_score = pred_score.item()
        pred_score = float(pred_score)

        return anomaly_map, pred_score

    def predict_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, float]]:
        results: List[Tuple[np.ndarray, float]] = []
        for i, img in enumerate(images):
            try:
                result = self.predict_single(img)
                results.append(result)
            except Exception as e:
                logger.error(f"批量推理第 {i} 张图片失败: {e}")
                h, w = img.shape[:2]
                empty_map = np.zeros((h, w), dtype=np.float32)
                results.append((empty_map, 0.0))
        return results

    def predict_tiles(
        self,
        tiles: List[np.ndarray],
        original_size: Tuple[int, int],
        tile_coords: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]]:
        if len(tiles) != len(tile_coords):
            raise ValueError("切片数量与坐标数量不匹配")

        results: List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]] = []
        for tile, coord in zip(tiles, tile_coords):
            anomaly_map, score = self.predict_single(tile)
            results.append((anomaly_map, score, coord))
        return results

    def assemble_anomaly_map(
        self,
        tile_results: List[Tuple[np.ndarray, float, Tuple[int, int, int, int]]],
        original_size: Tuple[int, int],
        tile_size: int,
        overlap: float = 0.25,
    ) -> np.ndarray:
        h, w = original_size[:2]
        full_map = np.zeros((h, w), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for anomaly_map, score, (x1, y1, x2, y2) in tile_results:
            tile_h = y2 - y1
            tile_w = x2 - x1
            resized_map = cv2.resize(
                anomaly_map, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR
            )

            full_map[y1:y2, x1:x2] += resized_map
            weight_map[y1:y2, x1:x2] += 1.0

        mask = weight_map > 0
        full_map[mask] /= weight_map[mask]

        return full_map
