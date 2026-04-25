from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

import gradio as gr
import numpy as np
from loguru import logger


_training_thread: Optional[threading.Thread] = None
_training_running = False
_training_log_lines: List[str] = []
_latest_csv: Optional[str] = None


def _create_train_tab():
    with gr.Row():
        with gr.Column(scale=1):
            dataset_dir = gr.Textbox(label="COCO 数据集目录", value="dataset/batch1_coco")
            model_select = gr.Dropdown(
                choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                value="yolov8n.pt",
                label="基础模型",
            )
            epochs = gr.Slider(10, 300, value=100, step=10, label="训练轮数")
            batch_size = gr.Slider(1, 32, value=8, step=1, label="批量大小")
            img_size = gr.Dropdown(choices=[320, 480, 640, 960], value=640, label="图片尺寸")
            with gr.Row():
                slice_enabled = gr.Checkbox(value=True, label="启用切片")
                slice_size = gr.Slider(256, 1024, value=640, step=64, label="切片大小")
            conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="推理置信度阈值")

            train_btn = gr.Button("开始训练", variant="primary")
            stop_btn = gr.Button("停止训练", variant="stop")

        with gr.Column(scale=2):
            train_status = gr.Textbox(label="训练状态", interactive=False)
            train_log = gr.Textbox(label="实时日志", lines=10, interactive=False, every=2)

    def start_training(ds, model, ep, bs, isz, sl_en, sl_sz):
        global _training_running, _training_log_lines
        if _training_running:
            return "训练正在进行中", "\n".join(_training_log_lines[-20:])

        _training_log_lines = [f"启动训练: {model}, epochs={ep}, batch={bs}"]

        def _train_worker():
            global _training_running
            try:
                _training_running = True
                from core.config import SlicerConfig, TrainingConfig
                from core.yolo_engine import PipelineA

                slicer_cfg = SlicerConfig(enabled=sl_en, slice_size=int(sl_sz)) if sl_en else SlicerConfig(enabled=False)
                config = TrainingConfig(
                    model=model,
                    dataset=ds,
                    epochs=int(ep),
                    batch_size=int(bs),
                    img_size=int(isz),
                    slicer=slicer_cfg,
                )

                pipeline = PipelineA(slicer_config=slicer_cfg)
                _training_log_lines.append("开始训练...")
                pipeline.train(config)
                _training_log_lines.append("训练完成!")

            except Exception as e:
                _training_log_lines.append(f"训练错误: {e}")
            finally:
                _training_running = False

        t = threading.Thread(target=_train_worker, daemon=True)
        t.start()
        return "训练已启动", "\n".join(_training_log_lines[-20:])

    def stop_training():
        global _training_running
        _training_running = False
        return "已请求停止", "\n".join(_training_log_lines[-20:])

    def refresh_log():
        return "\n".join(_training_log_lines[-30:])

    train_btn.click(start_training, [dataset_dir, model_select, epochs, batch_size, img_size, slice_enabled, slice_size], [train_status, train_log])
    stop_btn.click(stop_training, outputs=[train_status, train_log])
    train_log.attach_trigger_event(refresh_log, [train_log])


def _create_test_tab():
    with gr.Row():
        with gr.Column():
            model_path = gr.Textbox(label="模型路径", placeholder="runs/train_xxx/weights/best.pt")
            test_image = gr.Image(label="测试图片", type="filepath")
            test_conf = gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="置信度阈值")
            test_btn = gr.Button("推理", variant="primary")

        with gr.Column():
            result_image = gr.Image(label="检测结果")
            result_text = gr.Textbox(label="检测结果详情", lines=10, interactive=False)

    def run_test(img_path, model, conf):
        if not img_path or not model:
            return None, "请提供图片和模型路径"
        try:
            from core.yolo_engine import PipelineA

            pipeline = PipelineA()
            pipeline.load_model(model)
            result = pipeline.predict(img_path, conf)

            import cv2
            img = cv2.imread(img_path)
            for det in result.detections:
                h, w = img.shape[:2]
                x1, y1 = int(det.bbox.x1 * w), int(det.bbox.y1 * h)
                x2, y2 = int(det.bbox.x2 * w), int(det.bbox.y2 * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{det.class_name} {det.confidence:.2f}",
                           (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            lines = [f"检测到 {len(result.detections)} 个缺陷"]
            for det in result.detections:
                lines.append(f"  {det.class_name}: conf={det.confidence:.3f}, region={det.bbox.to_region_str()}")
            lines.append(f"\n耗时: {result.timing_ms}")

            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "\n".join(lines)
        except Exception as e:
            return None, f"推理失败: {e}"

    test_btn.click(run_test, [test_image, model_path, test_conf], [result_image, result_text])


def create_workbench_app():
    with gr.Blocks(title="工业缺陷检测 - 训练工作台") as app:
        gr.Markdown("# 工业缺陷检测 - 训练工作台")

        with gr.Tab("训练"):
            _create_train_tab()

        with gr.Tab("推理测试"):
            _create_test_tab()

    return app
