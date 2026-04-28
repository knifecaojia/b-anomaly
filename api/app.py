from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import router, set_pipeline
from core.config import SlicerConfig
from core.device import set_device
from core.yolo_engine import PipelineA


def _create_pipeline_b(
    anomalib_model_path: str,
    classifier_model_path: str = "",
    conf_threshold: float = 0.25,
    slicer_config: SlicerConfig | None = None,
):
    from core.config import ClassifierConfig
    from pipeline.pipeline_b import PipelineB

    cls_config = ClassifierConfig(
        type="yolo",
        model_path=classifier_model_path,
        conf_threshold=conf_threshold,
    )
    pipeline = PipelineB(
        classifier_config=cls_config,
        slicer_config=slicer_config,
    )
    pipeline.load_model(anomalib_model_path)
    return pipeline


def create_app(
    model_path: str,
    conf_threshold: float = 0.25,
    device: str = "auto",
    disable_slicing: bool = False,
    pipeline_type: str = "a",
    anomalib_model_path: str = "",
    classifier_model_path: str = "",
    rfdetr_model_variant: str = "s",
) -> FastAPI:
    logger.info(f"[准备] 初始化设备: {device}")
    set_device(device)

    slicer_config = None
    if not disable_slicing:
        slicer_config = SlicerConfig(enabled=True)
        logger.info(f"[准备] 切片已启用: size={slicer_config.slice_size}, overlap={slicer_config.overlap}")
    else:
        logger.info("[准备] 切片已禁用")

    if pipeline_type.lower() == "b":
        if not anomalib_model_path:
            raise ValueError("Pipeline B 需要指定 --anomalib-model 参数")
        pipeline = _create_pipeline_b(
            anomalib_model_path=anomalib_model_path,
            classifier_model_path=classifier_model_path,
            conf_threshold=conf_threshold,
            slicer_config=slicer_config,
        )
        logger.info(f"[准备] Pipeline B 模式 | 异常检测: {anomalib_model_path} | 分类器: {classifier_model_path or '无'}")
    elif pipeline_type.lower() == "c":
        from core.config import load_pipeline_c_config
        from pipeline.pipeline_c import PipelineC
        config = load_pipeline_c_config("config/pipeline_c.yaml")
        if slicer_config:
            config.slicer = slicer_config
        else:
            config.slicer.enabled = False
        config.training.model_variant = rfdetr_model_variant
        config.training.pretrain_weights = model_path
        pipeline = PipelineC(config=config)
        logger.info(f"[准备] Pipeline C 模式 | RF-DETR ({rfdetr_model_variant}) | 权重: {model_path}")
    else:
        pipeline = PipelineA(slicer_config=slicer_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline_type.lower() == "a":
            logger.info(f"[准备] 开始加载模型: {Path(model_path)}")
            pipeline.load_model(model_path)
        elif pipeline_type.lower() == "c":
            logger.info(f"[准备] 开始加载 Pipeline C 模型: {Path(model_path)}")
            pipeline.initialize()
        set_pipeline(pipeline, conf_threshold, pipeline_type=pipeline_type)
        logger.info(
            f"[准备] 模型就绪 | Pipeline {pipeline_type.upper()} | 类别: {pipeline.class_names} | 置信度阈值: {conf_threshold}"
        )
        yield
        logger.info("[准备] 服务关闭，释放资源")

    pipeline_desc = (
        "基于 YOLO 的工业产品缺陷检测服务"
        if pipeline_type.lower() == "a"
        else "基于 Anomalib PatchCore 异常检测 + YOLO 分类的工业缺陷检测服务"
        if pipeline_type.lower() == "b"
        else "基于 RF-DETR 端到端的工业缺陷检测服务"
    )

    tags_metadata = [
        {
            "name": "缺陷检测",
            "description": "产品缺陷检测核心接口，接收图片列表并返回检测结果",
        },
        {
            "name": "系统状态",
            "description": "服务健康检查、模型信息查询",
        },
    ]

    app = FastAPI(
        title="工业缺陷检测 API",
        summary=pipeline_desc,
        description=(
            "## 功能概述\n"
            "提供 RESTful 接口对工业产品图片进行缺陷检测，"
            "支持大图自动切片推理。\n\n"
            f"## 当前模式: Pipeline {pipeline_type.upper()}\n"
            + (
                "- Pipeline A: 基于 YOLO 的直接检测\n"
                "- 检测类别: 异物、过切\n"
                if pipeline_type.lower() == "a"
                else "- Pipeline B: Anomalib PatchCore 异常检测 + YOLO 分类\n"
                "- 异常检测: 零样本（仅用正常品训练）\n"
                "- 分类: 异物、过切\n"
            )
            + "\n## 快速开始\n"
            "1. 调用 `GET /health` 确认服务就绪\n"
            "2. 调用 `POST /get_latest_defect_infos` 提交检测请求\n"
        ),
        version="1.1.0",
        lifespan=lifespan,
        openapi_tags=tags_metadata,
        contact={"name": "缺陷检测团队"},
        license_info={"name": "内部使用"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    return app
