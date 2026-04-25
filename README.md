# 爱泼— 工业产品缺陷检测系统

基于 YOLO 的工业产品表面缺陷检测系统。支持模型训练、单图/批量推理、RESTful API 服务。

## 检测类别

| 类别     | 说明               |
| -------- | ------------------ |
| 异物     | 产品表面的外来物质 |
| 小NC过切 | 数控加工过切缺陷   |

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x（GPU 推理需要）
- NVIDIA GPU（推荐 6GB+ 显存）

### 安装

```bash
git clone <repo-url> && cd apple
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux
pip install -r requirements.txt
```

### 验证安装

```bash
python main.py --help
```

---

## 项目结构

```
apple/
├── main.py                  # 入口：CLI 命令分发
├── requirements.txt
├── config/
│   ├── default.yaml         # 全局默认配置
│   ├── pipeline_a.yaml      # 训练管线配置
│   └── defect_mapping.yaml  # 缺陷类别映射
├── core/                    # 核心模块
│   ├── config.py            # Pydantic 配置模型
│   ├── device.py            # GPU/CPU 设备管理
│   ├── timing.py            # 推理阶段计时器
│   ├── slicer.py            # 大图切片引擎
│   ├── dataset_manager.py   # 数据集管理
│   └── yolo_engine.py       # YOLO 推理管线
├── pipeline/
│   └── pipeline_a.py        # PipelineA 工厂
├── cli/
│   ├── train_cmd.py         # train / predict / benchmark 命令
│   └── predict_cmd.py       # serve / viewer 命令
├── api/                     # RESTful API
│   ├── app.py               # FastAPI 应用工厂
│   ├── routes.py            # 路由定义
│   ├── schemas.py           # 请求/响应模型
│   └── response_builder.py  # 响应构建
├── tests/                   # API 测试
│   ├── conftest.py
│   └── test_api.py
├── viewers/                 # 可视化工具
│   └── dataset_viewer.py    # Gradio 数据集浏览器
├── dataset/                 # 数据集（COCO / YOLO 格式）
├── runs/                    # 训练产出
└── logs/                    # 按日轮转日志
```

---

## 一、模型训练

### 1.1 数据集准备

数据集需要 YOLO 格式（`images/` + `labels/`），并包含 `dataset.yaml`：

```yaml
# dataset.yaml 示例
names:
  0: 异物
  1: 小NC过切
nc: 2
path: /absolute/path/to/dataset
train: train/images
val: val/images
test: test/images
```

项目已有数据集位于 `dataset/batch1_coco_yolo_direct/`，包含：

- 训练集：311 张图片
- 验证集：88 张图片
- 测试集：49 张图片
- 原始分辨率：5472 × 3648

### 1.2 训练命令

```bash
python main.py train --dataset <数据集路径> [选项]
```

**常用参数**：

| 参数             | 默认值         | 说明                                    |
| ---------------- | -------------- | --------------------------------------- |
| `--dataset`    | （必填）       | COCO/YOLO 数据集目录路径                |
| `--model`      | `yolov8n.pt` | 基础模型（yolov8n / yolov8m / yolo11m） |
| `--epochs`     | `100`        | 训练轮数                                |
| `--batch-size` | `16`         | 批量大小                                |
| `--img-size`   | `640`        | 输入图片尺寸                            |
| `--lr0`        | `0.01`       | 初始学习率                              |
| `--patience`   | `50`         | 早停耐心值（多少轮无改善后停止）        |
| `--device`     | `auto`       | 设备：auto / cpu / cuda                 |
| `--output-dir` | `runs`       | 输出目录                                |
| `--slice-size` | `640`        | 切片大小（像素）                        |
| `--overlap`    | `0.25`       | 切片重叠率                              |
| `--no-slice`   | `false`      | 禁用切片，直接缩放训练                  |

### 1.3 两种训练策略

**策略 A：直接训练（--no-slice）**

将 5472×3648 的高分辨率原图直接缩放到 640×640 训练。速度快、显存占用低，但小目标容易丢失。

```bash
python main.py train \
  --dataset dataset/batch1_coco_yolo_direct \
  --model yolov8n.pt \
  --epochs 100 \
  --batch-size 16 \
  --device cuda \
  --no-slice
```

**策略 B：切片训练（默认）**

将原图切成 640×640 的小片（25% 重叠），每张约 77 片。保留含缺陷切片，不含缺陷的按比例抽样作负样本。

```bash
python main.py train \
  --dataset dataset/batch1_coco_yolo_direct \
  --model yolov8n.pt \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

### 1.4 训练结果参考

在 RTX 3080 (10GB) 上，batch1 数据集的训练指标：

| 方案 | 模型    | 切片 | 整体 mAP50      | 整体 mAP50-95 | 训练耗时  |
| ---- | ------- | ---- | --------------- | ------------- | --------- |
| A    | YOLOv8n | 无   | 0.511           | 0.371         | ~7.5 分钟 |
| B    | YOLOv8n | 640  | **0.680** | 0.389         | ~24 分钟  |
| C    | YOLOv8m | 640  | 0.621           | 0.376         | ~35 分钟  |
| D    | YOLO11m | 640  | 0.629           | 0.368         | ~56 分钟  |

> 切片训练（方案 B）在当前数据集上效果最佳。详细报告见 `docs/训练验证报告_20260425.md`。

### 1.5 训练产出

训练完成后，结果保存在 `runs/detect/runs/train_<时间戳>/` 目录：

```
train_20260425_070520/
├── weights/
│   ├── best.pt          # 最佳权重（验证 mAP 最高）
│   └── last.pt          # 最后一轮权重
├── results.png          # 损失/mAP 曲线图
├── confusion_matrix.png # 混淆矩阵
├── results.csv          # 逐轮指标数据
└── args.yaml            # 实际训练参数
```

---

## 二、模型推理

### 2.1 单图/批量推理

```bash
python main.py predict --model <模型路径> --source <图片路径或目录> [选项]
```

| 参数             | 默认值    | 说明                    |
| ---------------- | --------- | ----------------------- |
| `--model`      | （必填）  | 模型权重文件路径（.pt） |
| `--source`     | （必填）  | 图片路径或目录          |
| `--conf`       | `0.25`  | 置信度阈值              |
| `--device`     | `auto`  | 设备                    |
| `--slice-size` | `640`   | 切片大小                |
| `--overlap`    | `0.25`  | 切片重叠率              |
| `--no-slice`   | `false` | 禁用切片                |

**示例**：

```bash
# 单张图片推理
python main.py predict \
  --model runs/detect/runs/train_20260425_070520/weights/best.pt \
  --source dataset/batch1_coco_yolo_direct/test/images/S0000045_P00_xxx.jpg

# 目录批量推理
python main.py predict \
  --model runs/detect/runs/train_20260425_070520/weights/best.pt \
  --source dataset/batch1_coco_yolo_direct/test/images

# 禁用切片（快速推理）
python main.py predict \
  --model runs/detect/runs/train_20260425_070520/weights/best.pt \
  --source dataset/batch1_coco_yolo_direct/test/images \
  --no-slice
```

**输出示例**：

```
图片: S0000045_P00_xxx.jpg
  耗时: {'加载图片': 84.2, '切片': 152.3, '推理': 1015.6, 'NMS合并': 2.1, '后处理': 0.3}
  [异物] conf=0.873 region=(0.1234,0.2345,0.3456,0.4567)
  [小NC过切] conf=0.721 region=(0.5678,0.6789,0.7890,0.8901)
```

### 2.2 性能基准测试

```bash
python main.py benchmark \
  --model runs/detect/runs/train_20260425_070520/weights/best.pt \
  --source dataset/batch1_coco_yolo_direct/test/images \
  --runs 10
```

---

## 三、RESTful API 服务

### 3.1 启动服务

```bash
python main.py serve --model <模型路径> [选项]
```

| 参数           | 默认值      | 说明             |
| -------------- | ----------- | ---------------- |
| `--model`    | （必填）    | 模型权重文件路径 |
| `--host`     | `0.0.0.0` | 监听地址         |
| `--port`     | `8000`    | 监听端口         |
| `--conf`     | `0.25`    | 默认置信度阈值   |
| `--device`   | `auto`    | 推理设备         |
| `--no-slice` | `false`   | 禁用切片推理     |

**示例**：

```bash
python main.py serve \
  --model runs/detect/runs/train_20260425_070520/weights/best.pt \
  --port 8000 \
  --conf 0.3
```

启动后访问：

- API 文档（Swagger）：`http://localhost:8000/docs`
- OpenAPI Schema：`http://localhost:8000/openapi.json`

### 3.2 API 接口

#### POST /get_latest_defect_infos — 缺陷检测

提交图片列表，返回检测结果。

**请求体**：

```json
{
  "job_id": "JOB20260425001",
  "sample_id": "S0000045",
  "position": "P00",
  "file_names": [
    "S0000045_P00_20260422100936_20260422100926.jpg",
    "S0000045_P03_20260422101134_20260422100926.jpg"
  ],
  "relative_dir": "dataset/batch1_coco_yolo_direct/test/images"
}
```

| 字段             | 类型     | 必填 | 说明                      |
| ---------------- | -------- | ---- | ------------------------- |
| `job_id`       | string   | 是   | 任务批次 ID               |
| `sample_id`    | string   | 是   | 样品 ID                   |
| `position`     | string   | 是   | 拍摄位置（P00/P03/P06）   |
| `file_names`   | string[] | 是   | 图片文件名列表，至少 1 个 |
| `relative_dir` | string   | 是   | 图片目录的相对路径        |

**响应体**：

```json
{
  "code": 200,
  "message": "success",
  "job_id": "JOB20260425001",
  "sample_id": "S0000045",
  "position": "P00",
  "relative_dir": "dataset/batch1_coco_yolo_direct/test/images",
  "timestamp": "1714023926000",
  "defect_infos": [
    {
      "file_name": "S0000045_P00_20260422100936_20260422100926.jpg",
      "defect_list": [
        {
          "type": "异物",
          "defect_infos": [
            {
              "region": "0.123456,0.234567,0.345678,0.456789",
              "conf": 0.87
            }
          ]
        }
      ]
    }
  ]
}
```

| 字段                                                   | 说明                       |
| ------------------------------------------------------ | -------------------------- |
| `code`                                               | 200=成功，500=模型未加载   |
| `defect_infos[].defect_list[].type`                  | 缺陷类型名称               |
| `defect_infos[].defect_list[].defect_infos[].region` | 归一化坐标 `x1,y1,x2,y2` |
| `defect_infos[].defect_list[].defect_infos[].conf`   | 置信度 0.0~1.0             |

#### GET /health — 健康检查

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_version": "best",
  "classes": ["异物", "小NC过切"],
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3080"
}
```

#### GET /model/info — 模型信息

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_path": "runs/detect/runs/train_20260425_070520/weights/best.pt",
  "classes": ["异物", "小NC过切"],
  "slicer_enabled": true,
  "slice_size": 640
}
```

### 3.3 调用示例

**Python（httpx）**：

```python
import httpx

resp = httpx.post("http://localhost:8000/get_latest_defect_infos", json={
    "job_id": "TEST001",
    "sample_id": "S0001",
    "position": "P00",
    "file_names": ["image1.jpg", "image2.jpg"],
    "relative_dir": "/path/to/images",
})
print(resp.json())
```

**curl**：

```bash
curl -X POST http://localhost:8000/get_latest_defect_infos \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "TEST001",
    "sample_id": "S0001",
    "position": "P00",
    "file_names": ["image1.jpg"],
    "relative_dir": "/path/to/images"
  }'
```

### 3.4 日志

- **控制台**：带完整日期时间戳和阶段标签 `[准备]` `[请求]` `[推理]` `[响应]`
- **文件**：按日保存在 `logs/app_YYYYMMDD.log`，默认保留 30 天
- 每个请求分配唯一 ID，贯穿全链路日志

```
2026-04-25 11:42:03 | INFO     | [准备] 模型就绪 | 类别: ['异物', '小NC过切'] | 置信度阈值: 0.25
2026-04-25 11:45:23 | INFO     | [请求] 2014a972 | 收到检测请求 | job_id=TEST001, 文件数=1
2026-04-25 11:45:23 | INFO     | [推理] 2014a972 | [1/1] 开始推理: image1.jpg
2026-04-25 11:45:24 | INFO     | [推理] 2014a972 | [1/1] 完成: 2个缺陷 | 加载图片=84ms -> 推理=1015ms
2026-04-25 11:45:24 | INFO     | [响应] 2014a972 | 请求完成 | 总耗时=1100ms
```

### 3.5 API 测试

```bash
# 先启动服务，再运行测试
python main.py serve --model runs/detect/runs/train_20260425_070520/weights/best.pt &

python -m pytest tests/test_api.py -v
```

---

## 四、其他工具

### 数据集浏览器

```bash
python main.py viewer --port 7860
```

基于 Gradio 的可视化工具，支持浏览数据集图片和标注。

### 配置文件

| 文件                           | 用途                                  |
| ------------------------------ | ------------------------------------- |
| `config/default.yaml`        | 全局默认配置（设备、切片、日志、API） |
| `config/pipeline_a.yaml`     | 训练管线参数（模型、增强、切片）      |
| `config/defect_mapping.yaml` | 缺陷类别映射与合并规则                |

---

## 技术栈

| 组件     | 技术                        | 说明                          |
| -------- | --------------------------- | ----------------------------- |
| 检测模型 | Ultralytics YOLOv8 / YOLO11 | 支持 n/s/m/l/x 多种规格       |
| 切片推理 | sahi                        | 大图自动切片 + NMS 合并       |
| Web 框架 | FastAPI + Uvicorn           | 异步高性能，自带 Swagger 文档 |
| 数据校验 | Pydantic v2                 | 请求/响应/配置模型            |
| 日志     | loguru                      | 按日轮转，带阶段标签          |
| 可视化   | Gradio                      | 数据集浏览器                  |
| 测试     | pytest + httpx              | API 端点测试                  |
