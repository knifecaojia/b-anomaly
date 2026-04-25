from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Generator

import httpx
import pytest

BASE_URL = "http://127.0.0.1:8000"
TEST_IMAGE_DIR = Path("dataset/batch1_coco_yolo_direct/test/images")
MODEL_PATH = "runs/detect/runs/train_20260425_070840/weights/best.pt"

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"


@pytest.fixture(scope="session")
def client() -> Generator[httpx.Client, None, None]:
    wait_for_server(timeout=60)
    transport = httpx.HTTPTransport(proxy=None)
    with httpx.Client(
        base_url=BASE_URL, timeout=120.0, transport=transport
    ) as c:
        yield c


@pytest.fixture(scope="session")
def test_images() -> list[dict]:
    if not TEST_IMAGE_DIR.exists():
        pytest.skip(f"测试图片目录不存在: {TEST_IMAGE_DIR}")
    images = sorted(TEST_IMAGE_DIR.glob("*.jpg"))
    if not images:
        pytest.skip("没有找到测试图片")
    return [{"dir": str(img.parent), "name": img.name} for img in images[:10]]


def wait_for_server(timeout: int = 60) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(f"{BASE_URL}/health", timeout=5.0, proxy=None)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    return
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(1)
    pytest.skip(f"服务在 {timeout}s 内未就绪，请先启动: python main.py serve --model {MODEL_PATH}")
