from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field, field_validator


def extract_position(file_name: str) -> str:
    m = re.search(r"_(P\d{2})_", file_name)
    return m.group(1) if m else ""


class DefectRequest(BaseModel):
    job_id: str = Field(description="作业ID，用于标识每次作业，要求唯一")
    sample_id: str = Field(description="样本ID，在同一作业下唯一")
    file_names: List[str] = Field(
        ...,
        min_length=1,
        description="待识别图片文件名列表。支持单点位多图输入",
    )
    relative_dir: str = Field(description="图片所在共享盘的相对目录")

    @field_validator("job_id", "sample_id", "relative_dir")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("字段不能为空")
        return v

    @field_validator("file_names")
    @classmethod
    def file_names_not_empty(cls, v: List[str]) -> List[str]:
        for fn in v:
            if not fn.strip():
                raise ValueError("file_names 中的元素不能为空")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "250330Day",
                    "sample_id": "S00005",
                    "file_names": [
                        "S00005_P00_20250330101712_XMTHBYPM9HV0000PUL21RB1L5.jpg",
                        "S00005_P03_20250330101723_XMTHBYPM9HV0000PUL21RB1L1.jpg",
                    ],
                    "relative_dir": "A/250330Day/Orignal/S00005/",
                }
            ]
        }
    }


class DefectDetail(BaseModel):
    region: str = Field(description="缺陷框区域，格式为 x1,y1,x2,y2 归一化坐标")
    conf: float = Field(ge=0.0, le=1.0, description="缺陷识别置信度，取值范围 0~1")


class DefectTypeResult(BaseModel):
    type: str = Field(description="缺陷类型，如 bent、fiber、scratch、insufficient_material")
    defect_infos: List[DefectDetail] = Field(
        default_factory=list, description="当前缺陷类型对应的识别明细列表"
    )


class ImageDefectResult(BaseModel):
    file_name: str = Field(description="图片文件名")
    defect_list: List[DefectTypeResult] = Field(
        default_factory=list, description="当前图片的缺陷分类结果列表"
    )


class DefectResponse(BaseModel):
    code: int = Field(description="接口返回状态码，200=成功，400=参数错误，500=系统内部错误")
    message: str = Field(description="接口返回信息，成功时为 success，失败时为具体错误原因")
    job_id: str = Field(default="", description="作业ID，与请求中的 job_id 一致")
    sample_id: str = Field(default="", description="样本ID，与请求中的 sample_id 一致")
    position: str = Field(default="", description="点位名称或点位编号，从文件名中自动提取")
    relative_dir: str = Field(default="", description="图片所在共享盘相对目录，与请求一致")
    timestamp: str = Field(default="", description="本次识别结果生成时间（毫秒级时间戳）")
    defect_infos: List[ImageDefectResult] = Field(
        default_factory=list, description="图片维度的识别结果列表，每张图片对应一个结果对象"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": 200,
                    "message": "success",
                    "job_id": "250330Day",
                    "sample_id": "S00005",
                    "position": "P00",
                    "relative_dir": "A/250330Day/Orignal/S00005/",
                    "timestamp": "1744205783127",
                    "defect_infos": [
                        {
                            "file_name": "S00005_P00_20250330101712.jpg",
                            "defect_list": [
                                {"type": "异物", "defect_infos": []},
                                {"type": "小NC过切", "defect_infos": []},
                            ],
                        }
                    ],
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    status: str = Field(description="服务状态: ok=就绪, no_model=模型未加载")
    model_version: str = Field(default="", description="模型版本（权重文件名）")
    classes: List[str] = Field(default_factory=list, description="模型支持的缺陷类别列表")
    device: str = Field(default="", description="推理设备: cuda 或 cpu")
    gpu_name: str = Field(default="", description="GPU 名称（使用 GPU 时）")
    pipeline_type: str = Field(default="A", description="当前 Pipeline 类型: A 或 B")


class ModelInfoResponse(BaseModel):
    model_path: str = Field(description="模型权重文件路径")
    classes: List[str] = Field(default_factory=list, description="缺陷类别列表")
    slicer_enabled: bool = Field(default=True, description="是否启用切片推理")
    slice_size: int = Field(default=640, description="切片尺寸（像素）")
    pipeline_type: str = Field(default="A", description="当前 Pipeline 类型: A 或 B")
