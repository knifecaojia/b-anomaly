from __future__ import annotations

import time
from typing import Dict, List

from api.schemas import (
    DefectDetail,
    DefectResponse,
    DefectTypeResult,
    ImageDefectResult,
    extract_position,
)
from pipeline import DetectionResult


def build_response(
    request_job_id: str,
    request_sample_id: str,
    request_relative_dir: str,
    file_results: Dict[str, List[DetectionResult]],
    class_names: List[str],
    first_file_name: str = "",
    code: int = 200,
    message: str = "success",
) -> DefectResponse:
    timestamp = str(int(time.time() * 1000))
    position = extract_position(first_file_name) if first_file_name else ""

    defect_infos: List[ImageDefectResult] = []
    for file_name, detections in file_results.items():
        class_dets: Dict[str, List[DefectDetail]] = {}
        for cn in class_names:
            class_dets[cn] = []

        for det in detections:
            cn = det.class_name
            if cn not in class_dets:
                class_dets[cn] = []
            region_str = f"{det.bbox.x1:.6f},{det.bbox.y1:.6f},{det.bbox.x2:.6f},{det.bbox.y2:.6f}"
            class_dets[cn].append(
                DefectDetail(region=region_str, conf=round(det.confidence, 6))
            )

        defect_list = [
            DefectTypeResult(type=cn, defect_infos=details)
            for cn, details in class_dets.items()
        ]

        defect_infos.append(
            ImageDefectResult(file_name=file_name, defect_list=defect_list)
        )

    return DefectResponse(
        code=code,
        message=message,
        job_id=request_job_id,
        sample_id=request_sample_id,
        position=position,
        relative_dir=request_relative_dir,
        timestamp=timestamp,
        defect_infos=defect_infos,
    )


def build_error_response(
    code: int,
    message: str,
    request_job_id: str = "",
    request_sample_id: str = "",
    request_position: str = "",
    request_relative_dir: str = "",
) -> DefectResponse:
    timestamp = str(int(time.time() * 1000))
    return DefectResponse(
        code=code,
        message=message,
        job_id=request_job_id,
        sample_id=request_sample_id,
        position=request_position,
        timestamp=timestamp,
        relative_dir=request_relative_dir,
    )
