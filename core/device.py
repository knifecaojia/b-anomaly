from __future__ import annotations

import torch
from loguru import logger


_cached_device: str | None = None


def get_device() -> str:
    global _cached_device
    if _cached_device is not None:
        return _cached_device

    if torch.cuda.is_available():
        _cached_device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"使用 GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        _cached_device = "cpu"
        logger.info("CUDA 不可用，使用 CPU")

    return _cached_device


def set_device(device: str) -> None:
    global _cached_device
    if device == "auto":
        _cached_device = None
        get_device()
    else:
        _cached_device = device
        logger.info(f"设备已设置为: {device}")


def get_device_info() -> dict:
    device = get_device()
    info = {"device": device}
    if device == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 1
        )
    return info
