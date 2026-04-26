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
        aug_enabled: bool = True,
        aug_rotation_degrees: int = 25,
        aug_crop_scale_min: float = 0.85,
        aug_hflip_prob: float = 0.5,
        aug_vflip_prob: float = 0.5,
    ) -> str:
        from anomalib.data import Folder
        from anomalib.deploy import ExportType
        from anomalib.engine import Engine
        from anomalib.models import Patchcore

        if layers is None:
            layers = ["layer2", "layer3"]

        train_augmentations = None
        if aug_enabled:
            from torchvision.transforms import v2

            train_augmentations = v2.Compose([
                v2.RandomRotation(degrees=aug_rotation_degrees),
                v2.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(aug_crop_scale_min, 1.0),
                ),
                v2.RandomHorizontalFlip(p=aug_hflip_prob),
                v2.RandomVerticalFlip(p=aug_vflip_prob),
            ])
            logger.info(
                f"训练增强已启用: 旋转±{aug_rotation_degrees}°, "
                f"裁剪缩放[{aug_crop_scale_min}, 1.0], "
                f"水平翻转{aug_hflip_prob:.0%}, 垂直翻转{aug_vflip_prob:.0%}"
            )

        data_dir = Path(data_dir)
        datamodule = Folder(
            name="apple_defects",
            root=str(data_dir),
            normal_dir="train/normal",
            abnormal_dir="test/abnormal",
            normal_test_dir="test/normal",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
        )

        model = Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_processor=Patchcore.configure_pre_processor(
                image_size=(image_size, image_size),
            ),
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
            logger.info(f"模型 checkpoint 已保存到: {ckpt_path}")

        try:
            engine.export(
                model=model,
                export_type=ExportType.TORCH,
            )
            export_dir = Path(output_dir) / "exported" / "torch"
            pt_files = list(export_dir.glob("*.pt")) if export_dir.exists() else []
            if pt_files:
                final_pt = Path(output_dir) / "patchcore.pt"
                shutil.copy2(str(pt_files[0]), str(final_pt))
                logger.info(f"Torch 模型已导出到: {final_pt}")
                self._ckpt_path = str(final_pt)
            else:
                logger.warning(f"导出目录中未找到 .pt 文件: {export_dir}")
        except Exception as e:
            logger.warning(f"Torch 导出失败: {e}")

        if not self._ckpt_path and ckpt_path.exists():
            self._ckpt_path = str(ckpt_path)

        self._model = model
        self._engine = engine
        return self._ckpt_path or ""

    def load_model(self, model_path: str) -> None:
        path = Path(model_path)

        if path.suffix == ".pt":
            self._load_torch_inferencer(str(path))
        elif path.suffix == ".ckpt":
            self._load_from_checkpoint(str(path))
        elif path.is_dir():
            pt_files = list(path.glob("*.pt"))
            ckpt_files = list(path.glob("*.ckpt"))
            if pt_files:
                self._load_torch_inferencer(str(pt_files[0]))
            elif ckpt_files:
                self._load_from_checkpoint(str(ckpt_files[0]))
            else:
                raise FileNotFoundError(f"目录中未找到模型文件: {model_path}")
        else:
            pt_path = path.with_suffix(".pt")
            if pt_path.exists():
                self._load_torch_inferencer(str(pt_path))
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

        self._ckpt_path = str(path)
        logger.info(f"Anomalib 模型已加载: {path}")

    def _load_torch_inferencer(self, pt_path: str) -> None:
        from anomalib.deploy import TorchInferencer

        self._inferencer = TorchInferencer(path=pt_path)

    def _load_from_checkpoint(self, ckpt_path: str) -> None:
        import torch
        from anomalib.models import Patchcore

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        hyper = ckpt.get("hyper_parameters", {})
        backbone = hyper.get("backbone", "wide_resnet50_2")
        layers = hyper.get("layers", ("layer2", "layer3"))
        coreset_ratio = hyper.get("coreset_sampling_ratio", 0.1)
        num_neighbors = hyper.get("num_neighbors", 9)

        model = Patchcore(
            backbone=backbone,
            layers=list(layers),
            coreset_sampling_ratio=coreset_ratio,
            num_neighbors=num_neighbors,
        )

        state_dict = ckpt.get("state_dict", {})
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("pre_processor.") and not k.startswith("post_processor.")}
        model.load_state_dict(filtered, strict=False)
        model.eval()

        if hasattr(model, "memory_bank") and "model" in ckpt:
            sub_model = ckpt["model"]
            if hasattr(sub_model, "memory_bank"):
                model.model.memory_bank = sub_model.memory_bank

        self._model = model

    def predict_single(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        if self._inferencer is not None:
            return self._predict_via_inferencer(image)
        elif self._model is not None:
            return self._predict_via_model(image)
        raise RuntimeError("模型未加载，请先调用 load_model()")

    def _predict_via_inferencer(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
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

    def _predict_via_model(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0)

        self._model.eval()
        with torch.no_grad():
            output = self._model(tensor)

        if hasattr(output, "anomaly_map"):
            amap = output.anomaly_map
            if hasattr(amap, "cpu"):
                amap = amap.cpu().numpy()
            score = float(output.pred_score) if hasattr(output, "pred_score") else float(amap.max())
            if isinstance(amap, np.ndarray) and amap.ndim >= 3:
                amap = amap.squeeze()
            return amap, score

        return np.zeros((256, 256), dtype=np.float32), 0.0

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
