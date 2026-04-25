from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def create_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="启动模型训练")
    parser.add_argument("--dataset", type=str, required=True, help="COCO 数据集目录路径")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="基础模型")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=16, help="批量大小")
    parser.add_argument("--img-size", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始学习率")
    parser.add_argument("--patience", type=int, default=50, help="早停耐心值")
    parser.add_argument("--device", type=str, default="auto", help="设备 (auto/cpu/cuda)")
    parser.add_argument("--output-dir", type=str, default="runs", help="输出目录")
    parser.add_argument("--slice-size", type=int, default=640, help="切片大小")
    parser.add_argument("--overlap", type=float, default=0.25, help="切片重叠率")
    parser.add_argument("--no-slice", action="store_true", help="禁用切片预处理")
    parser.add_argument("--config", type=str, default="config/pipeline_a.yaml", help="配置文件路径")


def run_train(args: argparse.Namespace) -> None:
    from core.config import load_training_config
    from core.device import set_device
    from core.yolo_engine import PipelineA

    set_device(args.device)

    overrides = {
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "lr0": args.lr0,
        "patience": args.patience,
        "device": args.device,
        "output_dir": args.output_dir,
    }
    config = load_training_config(args.config, overrides)

    if args.no_slice:
        config.slicer.enabled = False
    else:
        config.slicer.slice_size = args.slice_size
        config.slicer.overlap = args.overlap

    pipeline = PipelineA(slicer_config=config.slicer)
    pipeline.train(config)

    logger.info("训练完成")


def create_predict_parser(subparsers) -> None:
    parser = subparsers.add_parser("predict", help="运行模型推理")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="图片路径或目录")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--slice-size", type=int, default=640, help="切片大小")
    parser.add_argument("--overlap", type=float, default=0.25, help="切片重叠率")
    parser.add_argument("--no-slice", action="store_true", help="禁用切片")


def run_predict(args: argparse.Namespace) -> None:
    from core.config import SlicerConfig
    from core.device import set_device
    from core.yolo_engine import PipelineA

    set_device(args.device)

    slicer_config = None
    if not args.no_slice:
        slicer_config = SlicerConfig(
            enabled=True,
            slice_size=args.slice_size,
            overlap=args.overlap,
        )

    pipeline = PipelineA(slicer_config=slicer_config)
    pipeline.load_model(args.model)

    source = Path(args.source)
    if source.is_dir():
        images = sorted(
            list(source.glob("*.jpg"))
            + list(source.glob("*.png"))
            + list(source.glob("*.bmp"))
        )
    else:
        images = [source]

    for img_path in images:
        result = pipeline.predict(str(img_path), args.conf)
        print(f"\n图片: {img_path.name}")
        print(f"  耗时: {result.timing_ms}")
        if result.detections:
            for det in result.detections:
                print(
                    f"  [{det.class_name}] conf={det.confidence:.3f} "
                    f"region=({det.bbox.x1:.4f},{det.bbox.y1:.4f},"
                    f"{det.bbox.x2:.4f},{det.bbox.y2:.4f})"
                )
        else:
            print("  未检测到缺陷")


def create_benchmark_parser(subparsers) -> None:
    parser = subparsers.add_parser("benchmark", help="推理性能基准测试")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="测试图片目录")
    parser.add_argument("--runs", type=int, default=10, help="重复次数")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")


def run_benchmark(args: argparse.Namespace) -> None:
    import time

    from core.yolo_engine import PipelineA

    pipeline = PipelineA()
    pipeline.load_model(args.model)

    source = Path(args.source)
    images = sorted(
        list(source.glob("*.jpg")) + list(source.glob("*.png"))
    )[:10]

    if not images:
        logger.error("未找到测试图片")
        return

    all_times: list[float] = []
    for run in range(args.runs):
        start = time.perf_counter()
        for img in images:
            pipeline.predict(str(img), args.conf)
        elapsed = (time.perf_counter() - start) * 1000
        all_times.append(elapsed)
        print(f"第 {run+1}/{args.runs} 轮: {elapsed:.1f}ms ({len(images)} 张)")

    avg = sum(all_times) / len(all_times)
    per_img = avg / len(images)
    print(f"\n平均总耗时: {avg:.1f}ms, 单张平均: {per_img:.1f}ms")
