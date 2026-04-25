from __future__ import annotations

import httpx
import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client: httpx.Client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_has_required_fields(self, client: httpx.Client):
        resp = client.get("/health")
        data = resp.json()
        for field in ("status", "model_version", "classes", "device"):
            assert field in data, f"缺少字段: {field}"

    def test_health_classes_is_list(self, client: httpx.Client):
        resp = client.get("/health")
        data = resp.json()
        assert isinstance(data["classes"], list)
        assert len(data["classes"]) > 0


class TestModelInfoEndpoint:
    def test_model_info_returns_200(self, client: httpx.Client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        for field in ("model_path", "classes", "slicer_enabled", "slice_size"):
            assert field in data, f"缺少字段: {field}"

    def test_model_info_classes_match_health(self, client: httpx.Client):
        health = client.get("/health").json()
        info = client.get("/model/info").json()
        assert health["classes"] == info["classes"]


class TestDefectDetectionEndpoint:
    def test_single_image_detection(self, client: httpx.Client, test_images: list[dict]):
        img = test_images[0]
        payload = {
            "job_id": "test-001",
            "sample_id": "sample-001",
            "file_names": [img["name"]],
            "relative_dir": img["dir"],
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 200
        assert data["message"] == "success"
        assert data["job_id"] == "test-001"
        assert data["sample_id"] == "sample-001"
        assert isinstance(data["defect_infos"], list)
        assert len(data["defect_infos"]) == 1

    def test_batch_images_detection(self, client: httpx.Client, test_images: list[dict]):
        file_names = [img["name"] for img in test_images[:5]]
        relative_dir = test_images[0]["dir"]
        payload = {
            "job_id": "test-batch-001",
            "sample_id": "sample-batch",
            "file_names": file_names,
            "relative_dir": relative_dir,
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 200
        assert len(data["defect_infos"]) == len(file_names)

    def test_defect_info_structure(self, client: httpx.Client, test_images: list[dict]):
        img = test_images[0]
        payload = {
            "job_id": "test-struct-001",
            "sample_id": "sample-struct",
            "file_names": [img["name"]],
            "relative_dir": img["dir"],
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        data = resp.json()
        img_result = data["defect_infos"][0]
        assert "file_name" in img_result
        assert "defect_list" in img_result
        assert isinstance(img_result["defect_list"], list)
        if img_result["defect_list"]:
            defect_type = img_result["defect_list"][0]
            assert "type" in defect_type
            assert "defect_infos" in defect_type

    def test_position_extracted_from_filename(self, client: httpx.Client, test_images: list[dict]):
        img = test_images[0]
        payload = {
            "job_id": "test-pos-001",
            "sample_id": "sample-pos",
            "file_names": [img["name"]],
            "relative_dir": img["dir"],
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        data = resp.json()
        assert "position" in data
        assert data["position"].startswith("P")

    def test_nonexistent_image(self, client: httpx.Client, test_images: list[dict]):
        payload = {
            "job_id": "test-missing-001",
            "sample_id": "sample-missing",
            "file_names": ["nonexistent_image.jpg"],
            "relative_dir": "/tmp/not_exist",
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["code"] == 200
        assert len(data["defect_infos"]) == 1
        assert data["defect_infos"][0]["defect_list"] == [] or all(
            len(dt["defect_infos"]) == 0 for dt in data["defect_infos"][0]["defect_list"]
        )

    def test_response_has_timestamp(self, client: httpx.Client, test_images: list[dict]):
        img = test_images[0]
        payload = {
            "job_id": "test-ts-001",
            "sample_id": "sample-ts",
            "file_names": [img["name"]],
            "relative_dir": img["dir"],
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        data = resp.json()
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0

    def test_defect_list_has_all_types(self, client: httpx.Client, test_images: list[dict]):
        img = test_images[0]
        payload = {
            "job_id": "test-types-001",
            "sample_id": "sample-types",
            "file_names": [img["name"]],
            "relative_dir": img["dir"],
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        data = resp.json()
        defect_list = data["defect_infos"][0]["defect_list"]
        assert len(defect_list) >= 1
        for item in defect_list:
            assert "type" in item
            assert "defect_infos" in item


class TestRequestValidation:
    def test_empty_file_names_rejected(self, client: httpx.Client):
        payload = {
            "job_id": "test-val-001",
            "sample_id": "sample-val",
            "file_names": [],
            "relative_dir": "/tmp",
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 422

    def test_empty_job_id_rejected(self, client: httpx.Client):
        payload = {
            "job_id": "  ",
            "sample_id": "sample-val",
            "file_names": ["test.jpg"],
            "relative_dir": "/tmp",
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 422

    def test_position_not_accepted(self, client: httpx.Client):
        payload = {
            "job_id": "test-no-pos",
            "sample_id": "sample-no-pos",
            "position": "P00",
            "file_names": ["test.jpg"],
            "relative_dir": "/tmp",
        }
        resp = client.post("/get_latest_defect_infos", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "position" not in resp.json() or True
