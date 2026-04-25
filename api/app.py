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


def create_app(
    model_path: str,
    conf_threshold: float = 0.25,
    device: str = "auto",
    disable_slicing: bool = False,
) -> FastAPI:
    logger.info(f"[准备] 初始化设备: {device}")
    set_device(device)

    slicer_config = None
    if not disable_slicing:
        slicer_config = SlicerConfig(enabled=True)
        logger.info(f"[准备] 切片已启用: size={slicer_config.slice_size}, overlap={slicer_config.overlap}")
    else:
        logger.info("[准备] 切片已禁用")

    pipeline = PipelineA(slicer_config=slicer_config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(f"[准备] 开始加载模型: {Path(model_path)}")
        pipeline.load_model(model_path)
        set_pipeline(pipeline, conf_threshold)
        logger.info(
            f"[准备] 模型就绪 | 类别: {pipeline.class_names} | 置信度阈值: {conf_threshold}"
        )
        yield
        logger.info("[准备] 服务关闭，释放资源")

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
        summary="基于 YOLO 的工业产品缺陷检测服务",
        description=(
            "## 功能概述\n"
            "提供 RESTful 接口对工业产品图片进行缺陷检测，"
            "支持大图自动切片推理。\n\n"
            "## 检测类别\n"
            "- **异物**：产品表面的外来物质\n"
            "- **小NC过切**：加工过切缺陷\n\n"
            "## 快速开始\n"
            "1. 调用 `GET /health` 确认服务就绪\n"
            "2. 调用 `POST /get_latest_defect_infos` 提交检测请求\n"
        ),
        version="1.0.0",
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
