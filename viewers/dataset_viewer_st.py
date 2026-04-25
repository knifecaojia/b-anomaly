"""
工业缺陷检测 - 数据集查看器 (Streamlit 版)

替代原 Gradio 实现，解决 Gradio 5.x 启动超时和 API 兼容性问题。
功能：数据集浏览（标注可视化）、切片预览、统计总览。
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image as PILImage

# ──────────────────────── 颜色 ────────────────────────
_COLOR_PALETTE = [
    (255, 97, 0), (0, 200, 83), (30, 144, 255), (255, 215, 0),
    (186, 85, 211), (0, 206, 209), (255, 105, 180), (144, 238, 144),
    (240, 128, 128), (135, 206, 250), (255, 165, 0), (147, 112, 219),
]


# ──────────────────────── 工具函数 ────────────────────────
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


@st.cache_data(show_spinner=False)
def _load_coco(ann_path: str) -> Optional[dict]:
    try:
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载标注文件失败: {e}")
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
        (tw, th_), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        cv2.rectangle(img, (x, y - th_ - baseline - 4), (x + tw + 4, y), bgr, -1)
        cv2.putText(
            img, label, (x + 2, y - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
        )
    return img


def _bgr_to_pil(img: np.ndarray) -> PILImage.Image:
    """OpenCV BGR numpy -> PIL RGB Image"""
    return PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ──────────────────────── 数据加载 (缓存) ────────────────────────
@st.cache_data(show_spinner="正在加载数据集...")
def load_dataset(ds_dir: str, split: str):
    """返回 (coco, img_dir, cat_map, ann_index, stats_text) 或 (None, ..., error_msg)"""
    base = Path(ds_dir)
    ann_path = _find_annotation_file(base, split)
    if not ann_path:
        return None, None, None, None, f"找不到标注文件 ({base}/annotations/instances_{split}.json)"

    coco = _load_coco(str(ann_path))
    if not coco:
        return None, None, None, None, "标注文件解析失败"

    img_dir = _find_image_dir(base, split)
    if not img_dir:
        return None, None, None, None, "找不到图片目录"

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    label_counts = Counter()
    for ann in coco["annotations"]:
        label_counts[cat_map.get(ann["category_id"], "?")] += 1

    # 预构建 image_id → annotations 索引
    ann_index = defaultdict(list)
    for ann in coco["annotations"]:
        ann_index[ann["image_id"]].append(ann)

    stats_lines = [
        f"📁 标注文件: {ann_path}",
        f"🖼️ 图片目录: {img_dir}",
        f"📊 图片数: **{len(coco['images'])}**",
        f"🏷️ 标注数: **{len(coco['annotations'])}**",
        f"📂 类别数: **{len(coco['categories'])}**",
        "",
        "**各类标注数:**",
    ]
    for name, cnt in label_counts.most_common():
        stats_lines.append(f"- {name}: {cnt}")

    return coco, str(img_dir), cat_map, dict(ann_index), "\n".join(stats_lines)


# ──────────────────────── TAB 1: 数据集浏览 ────────────────────────
def render_browse_tab():
    col1, col2 = st.columns([3, 1])
    with col1:
        ds_dir = st.text_input("数据集根目录", value="dataset/batch1_coco", key="browse_dir")
    with col2:
        split = st.selectbox("数据集划分", ["train", "val", "test"], key="browse_split")

    if st.button("🔄 加载数据集", type="primary", key="load_btn"):
        st.session_state["dataset_loaded"] = True

    if not st.session_state.get("dataset_loaded"):
        st.info("👆 点击「加载数据集」按钮开始浏览")
        return

    coco, img_dir, cat_map, ann_index, stats = load_dataset(ds_dir, split)
    if coco is None:
        st.error(stats)
        return

    # --- 侧边栏：统计 + 筛选 ---
    left, right = st.columns([1, 3])

    with left:
        st.markdown(stats)
        st.divider()

        class_names = ["全部"] + sorted(set(cat_map.values()))
        selected_class = st.selectbox("按类别筛选", class_names, key="class_filter")

        # 根据筛选构建索引列表
        if selected_class == "全部":
            filtered_indices = list(range(len(coco["images"])))
        else:
            target_cat_ids = {cid for cid, name in cat_map.items() if name == selected_class}
            img_ids_with_class = set()
            for ann in coco["annotations"]:
                if ann["category_id"] in target_cat_ids:
                    img_ids_with_class.add(ann["image_id"])
            filtered_indices = [
                i for i, img in enumerate(coco["images"])
                if img["id"] in img_ids_with_class
            ]

        n = len(filtered_indices)
        st.metric("筛选结果", f"{n} 张")

    with right:
        if n == 0:
            st.warning("该类别下没有图片")
            return

        # 导航
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("⬅️ 上一张", key="prev"):
                if st.session_state.get("img_idx", 0) > 0:
                    st.session_state["img_idx"] -= 1
        with nav_col3:
            if st.button("下一张 ➡️", key="next"):
                if st.session_state.get("img_idx", 0) < n - 1:
                    st.session_state["img_idx"] += 1
        with nav_col2:
            idx = st.number_input(
                "当前索引",
                min_value=0,
                max_value=max(0, n - 1),
                value=min(st.session_state.get("img_idx", 0), n - 1),
                step=1,
                key="img_idx_input",
            )
            st.session_state["img_idx"] = idx

        # 显示图片
        real_idx = filtered_indices[idx]
        img_info = coco["images"][real_idx]
        img_path = Path(img_dir) / img_info["file_name"]

        if not img_path.exists():
            st.error(f"图片不存在: {img_path}")
            return

        img = cv2.imread(str(img_path))
        if img is None:
            st.error(f"无法读取图片: {img_path}")
            return

        orig_h, orig_w = img.shape[:2]
        display = _scale_image(img, max_dim=1280)
        disp_h, disp_w = display.shape[:2]
        scale = disp_w / orig_w

        img_anns = ann_index.get(img_info["id"], [])
        annotated = _draw_boxes(display, img_anns, cat_map, scale)
        pil_img = _bgr_to_pil(annotated)

        st.image(pil_img, caption=f"{img_info['file_name']}  ({orig_w}×{orig_h})  标注: {len(img_anns)}", width="stretch")


# ──────────────────────── TAB 2: 切片预览 ────────────────────────
def render_slice_tab():
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded = st.file_uploader("上传高分辨率图片", type=["jpg", "jpeg", "png", "bmp"], key="slice_upload")
        slice_size = st.slider("切片大小", 256, 2048, 640, step=64, key="slice_size")
        overlap = st.slider("重叠率", 0.0, 0.5, 0.25, step=0.05, key="slice_overlap")

        if uploaded is not None:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is not None:
                st.image(_bgr_to_pil(_scale_image(image, 400)), caption="原图预览", width="stretch")
            else:
                st.error("无法解码图片")
                return
        else:
            st.info("请上传图片")
            return

    with col2:
        try:
            from core.slicer import compute_slice_coords
        except ImportError:
            st.error("无法导入 core.slicer 模块")
            return

        h, w = image.shape[:2]
        coords = compute_slice_coords(w, h, int(slice_size), float(overlap))

        sample_size = min(16, len(coords))
        sampled = random.sample(coords, sample_size)

        st.markdown(
            f"原图尺寸: **{w} × {h}** · "
            f"切片大小: **{slice_size}** · "
            f"重叠率: **{overlap:.0%}** · "
            f"切片总数: **{len(coords)}** · "
            f"展示: **{sample_size}**"
        )

        # 网格展示切片
        cols = st.columns(4)
        for i, c in enumerate(sampled):
            crop = image[c.y_start:c.y_start + c.height, c.x_start:c.x_start + c.width]
            disp = _scale_image(crop, max_dim=320)
            ch, cw = disp.shape[:2]
            cv2.rectangle(disp, (0, 0), (cw - 1, ch - 1), (0, 200, 0), 1)
            cv2.putText(
                disp, f"({c.x_start},{c.y_start})",
                (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1,
            )
            with cols[i % 4]:
                st.image(_bgr_to_pil(disp), width="stretch")


# ──────────────────────── TAB 3: 统计总览 ────────────────────────
@st.cache_data(show_spinner="正在加载统计数据...")
def _compute_class_stats(ds_dir: str):
    """计算所有 split 的类别统计"""
    import pandas as pd

    base = Path(ds_dir)
    rows = []
    sample_images = []

    for split in ["train", "val", "test"]:
        ann_path = _find_annotation_file(base, split)
        img_dir = _find_image_dir(base, split)
        if not ann_path or not img_dir:
            continue

        coco = _load_coco(str(ann_path))
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
            rows.append({
                "类别": name,
                "划分": split,
                "标注数": cnt,
                "图片数": len(img_ids_per_class.get(name, set())),
            })

        # 采样缩略图
        sampled_imgs = random.sample(coco["images"], min(6, len(coco["images"])))
        for img_info in sampled_imgs:
            img_path = img_dir / img_info["file_name"]
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    disp = _scale_image(img, max_dim=200)
                    sample_images.append((_bgr_to_pil(disp), f"{split}/{img_info['file_name']}"))

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["类别", "划分", "标注数", "图片数"])
    return df, sample_images


def render_stats_tab():
    ds_dir = st.text_input("数据集根目录", value="dataset/batch1_coco", key="stats_dir")

    if st.button("📊 加载统计", type="primary", key="stats_btn"):
        st.session_state["stats_loaded"] = True

    if not st.session_state.get("stats_loaded"):
        st.info("👆 点击「加载统计」按钮查看数据集统计")
        return

    df, sample_images = _compute_class_stats(ds_dir)

    if df.empty:
        st.warning("未找到数据")
        return

    # 统计表
    st.subheader("各类别统计")
    st.dataframe(df, width="stretch", hide_index=True)

    # 可视化
    import altair as alt

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("类别:N", sort="-y"),
        y=alt.Y("标注数:Q"),
        color=alt.Color("划分:N"),
        tooltip=["类别", "划分", "标注数", "图片数"],
    ).properties(height=300)
    st.altair_chart(chart, width="stretch")

    # 样本展示
    if sample_images:
        st.subheader("随机样本预览")
        cols = st.columns(6)
        for i, (img, caption) in enumerate(sample_images):
            with cols[i % 6]:
                st.image(img, caption=caption, width="stretch")


# ──────────────────────── 主入口 ────────────────────────
def main():
    st.set_page_config(
        page_title="工业缺陷检测 - 数据集查看器",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(
        """
        <style>
        .stMainBlockContainer { padding-top: 1rem; }
        header[data-testid="stHeader"] { height: 2.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🔍 工业缺陷检测 - 数据集查看器")

    tab1, tab2, tab3 = st.tabs(["📂 数据集浏览", "🔪 切片预览", "📊 统计总览"])

    with tab1:
        render_browse_tab()
    with tab2:
        render_slice_tab()
    with tab3:
        render_stats_tab()


if __name__ == "__main__":
    main()
