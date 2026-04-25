from __future__ import annotations

import json
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import gradio as gr
import numpy as np
from loguru import logger

_COLOR_PALETTE = [
    (255, 97, 0), (0, 200, 83), (30, 144, 255), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 105, 180), (144, 238, 144),
    (240, 128, 128), (135, 206, 250), (255, 165, 0), (147, 112, 219),
]

_cache: Dict[str, dict] = {}


def _find_annotation_file(base_dir: Path, split: str) -> Optional[Path]:
    candidates = [
        base_dir / "annotations" / f"instances_{split}.json",
        base_dir / split / "annotations.json",
        base_dir / split / "annotations" / "instances.json",
        base_dir / "annotations" / f"{split}.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_image_dir(base_dir: Path, split: str) -> Optional[Path]:
    candidates = [
        base_dir / "images" / split,
        base_dir / split / "images",
        base_dir / "images",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            jpgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.bmp"))
            if jpgs:
                return p
    return None


def _load_coco(ann_path: Path) -> Optional[dict]:
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载标注文件失败: {e}")
        return None


def _scale_image(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _draw_boxes(
    image: np.ndarray,
    annotations: List[dict],
    cat_map: Dict[int, str],
    scale: float = 1.0,
) -> np.ndarray:
    img = image.copy()
    color_map = {}
    for idx, name in enumerate(sorted(set(cat_map.values()))):
        color_map[name] = _COLOR_PALETTE[idx % len(_COLOR_PALETTE)]

    for ann in annotations:
        bbox = ann["bbox"]
        x, y, w, h = [int(v * scale) for v in bbox]
        cat_name = cat_map.get(ann["category_id"], "?")
        color = color_map.get(cat_name, (0, 255, 0))
        bgr = (color[2], color[1], color[0])

        cv2.rectangle(img, (x, y), (x + w, y + h), bgr, 2)

        label = cat_name
        if "score" in ann:
            label += f" {ann['score']:.2f}"

        font_scale = 0.5 * scale if scale < 1 else 0.5
        thickness = max(1, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 4, y), bgr, -1)
        cv2.putText(
            img, label, (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )

    return img


def _build_dataset_tab():
    with gr.Row():
        dataset_dir = gr.Textbox(
            label="数据集根目录",
            placeholder="如: dataset/batch1_coco",
            value="dataset/batch1_coco",
        )
        split_select = gr.Dropdown(
            choices=["train", "val", "test"],
            value="train",
            label="数据集划分",
        )

    load_btn = gr.Button("加载数据集", variant="primary")

    with gr.Row():
        with gr.Column(scale=1):
            stats_text = gr.Textbox(label="数据集统计", lines=8, interactive=False)
            filter_class = gr.Dropdown(
                choices=["全部"],
                value="全部",
                label="按类别筛选",
            )
        with gr.Column(scale=3):
            with gr.Row():
                prev_btn = gr.Button("上一张", size="sm")
                image_idx = gr.Number(value=0, label="当前索引", precision=0)
                next_btn = gr.Button("下一张", size="sm")
                total_text = gr.Text(value="共 0 张", label="", show_label=False)
            image_display = gr.Image(label="标注可视化", height=700)

    cache_key_state = gr.State("")

    def load_dataset(ds_dir, split):
        empty_ret = (
            "请输入目录", gr.Dropdown(choices=["全部"], value="全部"),
            None, "", gr.Number(maximum=0, value=0, precision=0), "共 0 张",
        )
        if not ds_dir:
            return empty_ret

        base = Path(ds_dir)
        ann_path = _find_annotation_file(base, split)
        if not ann_path:
            return (
                f"找不到标注文件\n尝试过的路径:\n"
                f"  {base}/annotations/instances_{split}.json\n"
                f"  {base}/{split}/annotations.json",
                gr.Dropdown(choices=["全部"], value="全部"),
                None, "", gr.Number(maximum=0, value=0, precision=0), "共 0 张",
            )

        coco = _load_coco(ann_path)
        if not coco:
            return (
                "标注文件解析失败",
                gr.Dropdown(choices=["全部"], value="全部"),
                None, "", gr.Number(maximum=0, value=0, precision=0), "共 0 张",
            )

        img_dir = _find_image_dir(base, split)
        if not img_dir:
            return (
                "找不到图片目录",
                gr.Dropdown(choices=["全部"], value="全部"),
                None, "", gr.Number(maximum=0, value=0, precision=0), "共 0 张",
            )

        cat_map = {c["id"]: c["name"] for c in coco["categories"]}
        label_counts = Counter()
        for ann in coco["annotations"]:
            label_counts[cat_map.get(ann["category_id"], "?")] += 1

        lines = [
            f"标注文件: {ann_path}",
            f"图片目录: {img_dir}",
            f"图片数: {len(coco['images'])}",
            f"标注数: {len(coco['annotations'])}",
            f"类别数: {len(coco['categories'])}",
            "",
            "各类标注数:",
        ]
        for name, cnt in label_counts.most_common():
            lines.append(f"  {name}: {cnt}")

        class_choices = ["全部"] + [name for name, _ in label_counts.most_common()]

        # 预构建 image_id → annotations 索引，避免 show_image 时 O(N) 线性扫描
        ann_index = defaultdict(list)
        for ann in coco["annotations"]:
            ann_index[ann["image_id"]].append(ann)

        key = uuid.uuid4().hex
        _cache[key] = {
            "coco": coco,
            "img_dir": str(img_dir),
            "cat_map": cat_map,
            "ann_index": dict(ann_index),
            "filtered_indices": list(range(len(coco["images"]))),
        }
        n = len(coco["images"])

        return (
            "\n".join(lines),
            gr.Dropdown(choices=class_choices, value="全部"),
            None,
            key,
            gr.Number(maximum=max(0, n - 1), value=0, precision=0),
            f"共 {n} 张",
        )

    def filter_by_class(class_name, key):
        if not key or key not in _cache:
            return gr.Number(maximum=0, value=0, precision=0), "共 0 张"
        entry = _cache[key]
        coco = entry["coco"]
        cat_map = entry["cat_map"]
        if class_name == "全部":
            indices = list(range(len(coco["images"])))
        else:
            target_cat_ids = {cid for cid, name in cat_map.items() if name == class_name}
            img_ids_with_class = set()
            for ann in coco["annotations"]:
                if ann["category_id"] in target_cat_ids:
                    img_ids_with_class.add(ann["image_id"])
            indices = [
                i for i, img in enumerate(coco["images"])
                if img["id"] in img_ids_with_class
            ]
        entry["filtered_indices"] = indices
        n = len(indices)
        return gr.Number(maximum=max(0, n - 1), value=0, precision=0), f"共 {n} 张"

    def show_image(idx, key):
        if not key or key not in _cache:
            return None
        entry = _cache[key]
        coco = entry["coco"]
        if not coco or not coco.get("images"):
            return None
        filtered = entry.get("filtered_indices", [])
        idx = int(idx)
        if idx < 0 or idx >= len(filtered):
            return None

        real_idx = filtered[idx]
        img_info = coco["images"][real_idx]
        img_path = Path(entry["img_dir"]) / img_info["file_name"]
        if not img_path.exists():
            return None

        img = cv2.imread(str(img_path))
        if img is None:
            return None

        orig_h, orig_w = img.shape[:2]
        display = _scale_image(img, max_dim=1280)
        disp_h, disp_w = display.shape[:2]
        scale = disp_w / orig_w

        cat_map = entry.get("cat_map", {})
        ann_index = entry.get("ann_index", {})
        img_anns = ann_index.get(img_info["id"], [])
        annotated = _draw_boxes(display, img_anns, cat_map, scale)

        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def nav_prev(idx):
        return max(0, int(idx) - 1)

    def nav_next(idx, key):
        if not key or key not in _cache:
            return int(idx)
        filtered = _cache[key].get("filtered_indices", [])
        return min(len(filtered) - 1, int(idx) + 1)

    load_btn.click(
        load_dataset,
        [dataset_dir, split_select],
        [stats_text, filter_class, image_display, cache_key_state, image_idx, total_text],
    )
    filter_class.change(
        filter_by_class,
        [filter_class, cache_key_state],
        [image_idx, total_text],
    )
    image_idx.change(show_image, [image_idx, cache_key_state], [image_display])
    prev_btn.click(nav_prev, [image_idx], [image_idx])
    next_btn.click(nav_next, [image_idx, cache_key_state], [image_idx])


def _build_slice_preview_tab():
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="上传高分辨率图片", type="numpy", height=400)
            with gr.Row():
                slice_size = gr.Slider(256, 2048, value=640, step=64, label="切片大小")
                overlap = gr.Slider(0.0, 0.5, value=0.25, step=0.05, label="重叠率")
            slice_btn = gr.Button("预览切片", variant="primary")
            slice_info = gr.Textbox(label="切片信息", interactive=False)
        with gr.Column(scale=2):
            slice_gallery = gr.Gallery(label="随机切片预览", columns=4, height=700)

    def preview_slices(image, sz, ov):
        if image is None:
            return [], "请先上传图片"
        from core.slicer import compute_slice_coords

        h, w = image.shape[:2]
        coords = compute_slice_coords(w, h, int(sz), float(ov))

        sample_size = min(16, len(coords))
        sampled = random.sample(coords, sample_size)

        gallery = []
        for c in sampled:
            crop = image[
                c.y_start : c.y_start + c.height,
                c.x_start : c.x_start + c.width,
            ]
            disp = _scale_image(crop, max_dim=320)
            ch, cw = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (cw - 1, ch - 1), (0, 200, 0), 1)
            cv2.putText(
                disp,
                f"({c.x_start},{c.y_start})",
                (3, 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 200, 0),
                1,
            )
            gallery.append(disp)

        info = (
            f"原图尺寸: {w} x {h}\n"
            f"切片大小: {int(sz)}\n"
            f"重叠率: {float(ov):.0%}\n"
            f"切片总数: {len(coords)}\n"
            f"展示数量: {sample_size}"
        )
        return gallery, info

    slice_btn.click(
        preview_slices,
        [image_input, slice_size, overlap],
        [slice_gallery, slice_info],
    )


def _build_class_stats_tab():
    with gr.Row():
        dataset_dir2 = gr.Textbox(
            label="数据集根目录",
            value="dataset/batch1_coco",
        )
        load_stats_btn = gr.Button("加载统计", variant="primary")

    stats_display = gr.Dataframe(
        headers=["类别", "标注数", "图片数（含该类）"],
        label="各类别统计",
        interactive=False,
    )
    sample_gallery = gr.Gallery(label="各类样本图片", columns=6, height=500)

    def load_class_stats(ds_dir):
        if not ds_dir:
            return [], []
        base = Path(ds_dir)
        rows = []
        all_samples = []

        for split in ["train", "val", "test"]:
            ann_path = _find_annotation_file(base, split)
            img_dir = _find_image_dir(base, split)
            if not ann_path or not img_dir:
                continue

            coco = _load_coco(ann_path)
            if not coco:
                continue

            cat_map = {c["id"]: c["name"] for c in coco["categories"]}
            label_counts = Counter()
            img_ids_per_class: Dict[str, set] = {}

            for ann in coco["annotations"]:
                name = cat_map.get(ann["category_id"], "?")
                label_counts[name] += 1
                img_ids_per_class.setdefault(name, set()).add(ann["image_id"])

            for name, cnt in label_counts.most_common():
                rows.append([f"{name} ({split})", cnt, len(img_ids_per_class.get(name, set()))])

            sampled_imgs = random.sample(coco["images"], min(6, len(coco["images"])))
            for img_info in sampled_imgs:
                img_path = img_dir / img_info["file_name"]
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        disp = _scale_image(img, max_dim=200)
                        all_samples.append(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))

        return rows, all_samples

    load_stats_btn.click(load_class_stats, [dataset_dir2], [stats_display, sample_gallery])


def create_viewer_app():
    with gr.Blocks(title="工业缺陷检测 - 数据集查看器") as app:
        gr.Markdown("# 工业缺陷检测 - 数据集查看器")

        with gr.Tab("数据集浏览"):
            _build_dataset_tab()

        with gr.Tab("切片预览"):
            _build_slice_preview_tab()

        with gr.Tab("统计总览"):
            _build_class_stats_tab()

    return app
