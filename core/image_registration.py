from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger


class ImageRegistration:
    def __init__(
        self,
        max_features: int = 5000,
        good_match_percent: float = 0.15,
        min_matches: int = 10,
        method: str = "sift",
    ) -> None:
        self._max_features = max_features
        self._good_match_percent = good_match_percent
        self._min_matches = min_matches
        self._method = method.lower()
        if self._method == "sift":
            self._detector = cv2.SIFT_create(nfeatures=max_features)
            flann_params = dict(algorithm=1, trees=5)
            self._matcher = cv2.FlannBasedMatcher(flann_params, {})
        else:
            self._detector = cv2.ORB_create(max_features)
            self._matcher = cv2.DescriptorMatcher_create(
                cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
            )
        self._template: Optional[np.ndarray] = None
        self._template_gray: Optional[np.ndarray] = None
        self._template_kp = None
        self._template_desc = None

    @property
    def template(self) -> Optional[np.ndarray]:
        return self._template

    @property
    def is_ready(self) -> bool:
        return self._template is not None

    def set_template(self, template: np.ndarray) -> None:
        self._template = template.copy()
        self._template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self._template_kp, self._template_desc = self._detector.detectAndCompute(
            self._template_gray, None
        )
        logger.info(
            f"配准模板已设置: {template.shape[1]}x{template.shape[0]}, "
            f"特征点={len(self._template_kp)}"
        )

    def load_template(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"无法读取模板图片: {path}")
        self.set_template(img)
        logger.info(f"配准模板已加载: {path}")

    def save_template(self, path: str) -> None:
        if self._template is None:
            raise RuntimeError("模板未设置")
        cv2.imwrite(path, self._template)
        logger.info(f"配准模板已保存: {path}")

    def register(
        self,
        image: np.ndarray,
        warp_mode: int = cv2.MOTION_AFFINE,
        partial_affine: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if self._template is None:
            raise RuntimeError("模板未设置，请先调用 set_template()")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = self._detector.detectAndCompute(gray, None)

        if desc is None or len(kp) < self._min_matches:
            logger.warning(f"特征点不足({len(kp)}), 返回原图")
            return image, np.eye(2, 3, dtype=np.float64), 0.0

        good_matches = self._match_features(desc, kp)

        if len(good_matches) < self._min_matches:
            logger.warning(f"匹配点不足({len(good_matches)}), 返回原图")
            return image, np.eye(2, 3, dtype=np.float64), 0.0

        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [self._template_kp[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        if partial_affine:
            # Similarity transform: rotation + uniform scale + translation only
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts, None, cv2.RANSAC, ransacReprojThreshold=3.0
            )
        else:
            matrix, inliers = cv2.estimateAffine2D(
                src_pts, dst_pts, None, cv2.RANSAC, ransacReprojThreshold=3.0
            )

        if matrix is None:
            logger.warning("仿射矩阵估计失败, 返回原图")
            return image, np.eye(2, 3, dtype=np.float64), 0.0

        inlier_count = int(inliers.sum()) if inliers is not None else 0
        match_ratio = inlier_count / len(good_matches)

        h, w = self._template_gray.shape[:2]
        aligned = cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR
        )

        return aligned, matrix, match_ratio

    def _match_features(self, desc, kp) -> list:
        if self._method == "sift":
            return self._match_sift(desc)
        return self._match_orb(desc)

    def _match_sift(self, desc) -> list:
        knn_matches = self._matcher.knnMatch(desc, self._template_desc, k=2)
        good = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        logger.debug(f"SIFT: {len(knn_matches)} 组匹配, Lowe筛选后 {len(good)} 个")
        return good

    def _match_orb(self, desc) -> list:
        matches = self._matcher.match(desc, self._template_desc)
        matches = sorted(matches, key=lambda x: x.distance)
        num_good = max(int(len(matches) * self._good_match_percent), self._min_matches)
        return matches[:num_good]

    def register_batch(
        self,
        images: list[np.ndarray],
    ) -> list[Tuple[np.ndarray, float]]:
        results: list[Tuple[np.ndarray, float]] = []
        for i, img in enumerate(images):
            aligned, _, ratio = self.register(img)
            results.append((aligned, ratio))
        return results

    def generate_template_from_images(
        self,
        images: list[np.ndarray],
        method: str = "median",
    ) -> np.ndarray:
        if not images:
            raise ValueError("图片列表不能为空")

        reference = images[0]
        self.set_template(reference)

        aligned_list = [reference]
        for i, img in enumerate(images[1:], 1):
            aligned, _, _ = self.register(img)
            aligned_list.append(aligned)

        if method == "median":
            template = self._incremental_median(aligned_list)
        else:
            template = self._incremental_mean(aligned_list)

        self.set_template(template)
        logger.info(
            f"模板已从 {len(images)} 张图片生成 (method={method})"
        )
        return template

    def _incremental_mean(self, images: list[np.ndarray]) -> np.ndarray:
        acc = np.zeros_like(images[0], dtype=np.float64)
        for img in images:
            acc += img.astype(np.float64)
        return (acc / len(images)).astype(np.uint8)

    def _incremental_median(self, images: list[np.ndarray]) -> np.ndarray:
        return self._incremental_mean(images)

    def generate_template_from_dir(
        self,
        image_dir: str,
        max_images: int = 50,
        method: str = "median",
    ) -> np.ndarray:
        dir_path = Path(image_dir)
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        files = sorted(
            [f for f in dir_path.rglob("*") if f.suffix.lower() in IMAGE_EXTS]
        )

        if not files:
            raise FileNotFoundError(f"目录中未找到图片: {image_dir}")

        selected = files[:max_images]
        images = []
        for f in selected:
            img = cv2.imread(str(f))
            if img is not None:
                images.append(img)

        if not images:
            raise RuntimeError(f"无法读取任何图片从: {image_dir}")

        logger.info(f"从 {len(images)} 张图片生成模板 (目录: {image_dir})")
        return self.generate_template_from_images(images, method)
