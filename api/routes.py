from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.response_builder import build_error_response, build_response
from api.schemas import DefectRequest, DefectResponse, HealthResponse, ModelInfoResponse
from core.device import get_device_info
from core.yolo_engine import PipelineA
from pipeline import BasePipeline

router = APIRouter()

_pipeline: BasePipeline | None = None
_conf_threshold: float = 0.25
_pipeline_type: str = "a"


def set_pipeline(pipeline: BasePipeline, conf_threshold: float = 0.25, pipeline_type: str = "a") -> None:
    global _pipeline, _conf_threshold, _pipeline_type
    _pipeline = pipeline
    _conf_threshold = conf_threshold
    _pipeline_type = pipeline_type


@router.post(
    "/get_latest_defect_infos",
    response_model=DefectResponse,
    tags=["缺陷检测"],
    summary="提交缺陷检测请求",
    description=(
        "接收一组图片文件名和目录路径，对每张图片执行缺陷检测，"
        "返回按文件和缺陷类型分组的结果。\n\n"
        "**处理流程**: 接收请求 → 校验图片路径 → 推理 → 返回结果\n\n"
        f"**当前模式**: Pipeline {_pipeline_type.upper()}\n\n"
        "**支持大图自动切片**: 当图片尺寸超过 2 倍切片尺寸时，"
        "自动切片推理后合并结果。"
    ),
    responses={
        200: {
            "description": "检测完成（包括模型未加载时的错误响应）",
            "content": {
                "application/json": {
                    "example": {
                        "code": 200,
                        "message": "success",
                        "job_id": "JOB001",
                        "defect_infos": [],
                    }
                }
            },
        },
        422: {"description": "请求参数校验失败"},
    },
)
async def detect_defects(request: DefectRequest) -> DefectResponse:
    request_id = uuid.uuid4().hex[:8]
    total_start = time.perf_counter()
    logger.info(
        f"[请求] {request_id} | 收到检测请求 | "
        f"job_id={request.job_id}, 文件数={len(request.file_names)}, "
        f"目录={request.relative_dir}"
    )

    if _pipeline is None:
        logger.error(f"[请求] {request_id} | 模型未加载，无法处理请求")
        return build_error_response(
            500, "模型未加载",
            request.job_id, request.sample_id, "", request.relative_dir,
        )

    engine_check = None
    if isinstance(_pipeline, PipelineA):
        engine_check = _pipeline._engine.is_loaded
    else:
        engine_check = _pipeline.is_loaded

    if not engine_check:
        logger.error(f"[请求] {request_id} | 模型未加载，无法处理请求")
        return build_error_response(
            500, "模型未加载",
            request.job_id, request.sample_id, "", request.relative_dir,
        )

    file_results: Dict[str, list] = {}
    for i, file_name in enumerate(request.file_names, 1):
        img_path = Path(request.relative_dir) / file_name
        if not img_path.exists():
            logger.warning(
                f"[推理] {request_id} | [{i}/{len(request.file_names)}] "
                f"图片不存在: {img_path}"
            )
            file_results[file_name] = []
            continue

        logger.info(
            f"[推理] {request_id} | [{i}/{len(request.file_names)}] "
            f"开始推理: {file_name}"
        )
        try:
            prediction = _pipeline.predict(str(img_path), _conf_threshold)
            file_results[file_name] = prediction.detections

            total_defects = len(prediction.detections)
            timing_summary = " -> ".join(
                f"{k}={v:.0f}ms" for k, v in prediction.timing_ms.items()
            )
            logger.info(
                f"[推理] {request_id} | [{i}/{len(request.file_names)}] "
                f"完成: {total_defects}个缺陷 | {timing_summary}"
            )
        except Exception as e:
            logger.error(f"[推理] {request_id} | [{i}/{len(request.file_names)}] 推理失败: {e}")
            file_results[file_name] = []

    response = build_response(
        request_job_id=request.job_id,
        request_sample_id=request.sample_id,
        request_relative_dir=request.relative_dir,
        file_results=file_results,
        class_names=_pipeline.class_names,
        first_file_name=request.file_names[0] if request.file_names else "",
    )

    elapsed_ms = (time.perf_counter() - total_start) * 1000
    total_defects = sum(len(d) for d in file_results.values())
    logger.info(
        f"[响应] {request_id} | 请求完成 | "
        f"总耗时={elapsed_ms:.0f}ms, 总缺陷数={total_defects}, "
        f"文件数={len(request.file_names)}"
    )

    return response


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["系统状态"],
    summary="健康检查",
    description="查询服务健康状态、模型加载情况和推理设备信息。可用于心跳检测和负载均衡。",
)
async def health_check() -> HealthResponse:
    device_info = get_device_info()
    classes = _pipeline.class_names if _pipeline else []

    model_ver = ""
    if isinstance(_pipeline, PipelineA) and _pipeline._engine._model_path:
        model_ver = Path(_pipeline._engine._model_path).stem
    elif _pipeline and hasattr(_pipeline, "_ckpt_path") and _pipeline._ckpt_path:
        model_ver = Path(_pipeline._ckpt_path).stem

    if isinstance(_pipeline, PipelineA):
        loaded = _pipeline._engine.is_loaded
    else:
        loaded = _pipeline.is_loaded if _pipeline else False

    return HealthResponse(
        status="ok" if loaded else "no_model",
        model_version=model_ver,
        classes=classes,
        device=device_info.get("device", "unknown"),
        gpu_name=device_info.get("gpu_name", ""),
        pipeline_type=_pipeline_type.upper(),
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["系统状态"],
    summary="查询模型信息",
    description="获取当前加载模型的详细信息，包括权重路径、支持类别和切片配置。",
    responses={
        200: {"description": "模型信息"},
        503: {"description": "模型未加载"},
    },
)
async def model_info() -> ModelInfoResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    model_path = ""
    slicer_enabled = True
    slice_size = 640

    if isinstance(_pipeline, PipelineA):
        model_path = _pipeline._engine._model_path or ""
        slicer_enabled = _pipeline._slicer_config.enabled
        slice_size = _pipeline._slicer_config.slice_size
    else:
        if hasattr(_pipeline, "_ckpt_path") and _pipeline._ckpt_path:
            model_path = _pipeline._ckpt_path
        slicer_enabled = _pipeline._slicer_config.enabled
        slice_size = _pipeline._slicer_config.slice_size

    return ModelInfoResponse(
        model_path=model_path,
        classes=_pipeline.class_names,
        slicer_enabled=slicer_enabled,
        slice_size=slice_size,
        pipeline_type=_pipeline_type.upper(),
    )
