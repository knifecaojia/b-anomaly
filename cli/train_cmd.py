from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def create_train_parser(subparsers) -> None:
    parser = subparsers.add_parser("train", help="启动模型训练")
    parser.add_argument("--pipeline", type=str, default="a", choices=["a", "b"], help="Pipeline 类型: a(YOLO) 或 b(Anomalib PatchCore)")
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
    parser.add_argument("--config", type=str, default="", help="配置文件路径（留空则根据 pipeline 类型自动选择）")
    parser.add_argument("--ok-dir", type=str, default="", help="Pipeline B: OK图目录路径")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2", help="Pipeline B: PatchCore 骨干网络")


def run_train(args: argparse.Namespace) -> None:
    from core.config import load_training_config
    from core.device import set_device

    set_device(args.device)

    if args.pipeline.lower() == "b":
        _run_train_pipeline_b(args)
    else:
        _run_train_pipeline_a(args)


def _run_train_pipeline_a(args: argparse.Namespace) -> None:
    from core.yolo_engine import PipelineA

    config_path = args.config or "config/pipeline_a.yaml"
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
    config = load_training_config(config_path, overrides)

    if args.no_slice:
        config.slicer.enabled = False
    else:
        config.slicer.slice_size = args.slice_size
        config.slicer.overlap = args.overlap

    pipeline = PipelineA(slicer_config=config.slicer)
    pipeline.train(config)

    logger.info("Pipeline A 训练完成")


def _run_train_pipeline_b(args: argparse.Namespace) -> None:
    from core.config import load_pipeline_b_config
    from pipeline.pipeline_b import PipelineB

    config_path = args.config or "config/pipeline_b.yaml"
    config = load_pipeline_b_config(config_path)

    anomalib_data_dir = args.dataset
    if anomalib_data_dir and Path(anomalib_data_dir).exists():
        coco_dir = Path(anomalib_data_dir)
        if (coco_dir / "annotations").exists() or (coco_dir / "train").exists():
            from dataset.prepare_anomalib_data import prepare_dataset
            ok_dir = args.ok_dir if args.ok_dir else None
            anomalib_data_dir = prepare_dataset(
                ok_dir=ok_dir,
                coco_dir=str(coco_dir),
                output_dir=str(coco_dir.parent / "anomalib_data"),
            )
            logger.info(f"Anomalib 数据已准备: {anomalib_data_dir}")

    pipeline = PipelineB(anomalib_config=config.anomalib, slicer_config=config.slicer)

    pipeline._engine.train(
        data_dir=anomalib_data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone or config.anomalib.backbone,
        layers=config.anomalib.layers,
        coreset_sampling_ratio=config.anomalib.coreset_sampling_ratio,
        num_neighbors=config.anomalib.num_neighbors,
        image_size=config.anomalib.image_size,
        train_batch_size=config.anomalib.train_batch_size,
        eval_batch_size=config.anomalib.eval_batch_size,
        num_workers=config.anomalib.num_workers,
    )

    logger.info("Pipeline B 训练完成")


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
