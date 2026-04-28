#  工业产品缺陷检测系统

基于 RF-DETR 的工业产品表面缺陷检测系统。支持模型训练、单图/批量推理、RESTful API 服务、Docker 容器化部署。

**项目地址**：<https://github.com/knifecaojia/b-anomaly>

## 检测类别

| 类别     | 说明                 |
| ------ | ------------------ |
| defect | 统一缺陷检测（异物 + 小NC过切） |

> 当前模型采用「二分类合并」策略，将异物和小NC过切统一识别为 defect。F1 综合得分 89.5%（精确率 85%，召回率 94.4%）。

## 快速开始

### 环境要求

- Python 3.10+
- CUDA 12.x（GPU 推理需要）
- NVIDIA GPU（推荐 10GB+ 显存）

### 安装

```bash
git clone https://github.com/knifecaojia/b-anomaly.git && cd b-anomaly
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux
pip install -r requirements.txt
```

### 验证安装

```bash
python main.py --help
```

***

## 项目结构

```
apple/
├── main.py                     # 入口：CLI 命令分发
├── requirements.txt
├── Dockerfile                  # 容器化部署（内嵌模型）
├── models/                     # 推理模型权重
│   ├── pipeline_c_rfdetr_medium_best_ema.pth   # RF-DETR Medium (默认)
│   └── pipeline_a_yolov8n_best.pt              # YOLOv8n
├── config/
│   ├── default.yaml            # 全局默认配置
│   ├── pipeline_a.yaml         # Pipeline A 训练配置
│   ├── pipeline_b.yaml         # Pipeline B 训练配置
│   ├── pipeline_c.yaml         # Pipeline C 训练配置
│   └── defect_mapping.yaml     # 缺陷类别映射
├── core/                       # 核心模块
│   ├── config.py               # Pydantic 配置模型
│   ├── device.py               # GPU/CPU 设备管理
│   ├── timing.py               # 推理阶段计时器
│   ├── slicer.py               # 大图切片引擎
│   ├── image_registration.py   # SIFT 图像配准
│   ├── dataset_manager.py      # 数据集管理
│   ├── yolo_engine.py          # YOLO 推理引擎 (Pipeline A)
│   ├── anomalib_engine.py      # Anomalib 引擎 (Pipeline B)
│   ├── rfdetr_engine.py        # RF-DETR 推理引擎 (Pipeline C)
│   ├── defect_classifier.py    # 缺陷分类器
│   ├── anomaly_region_extractor.py  # 异常区域提取
│   └── ...
├── pipeline/
│   ├── __init__.py             # 基类与数据结构
│   ├── pipeline_a.py           # Pipeline A (YOLO)
│   ├── pipeline_b.py           # Pipeline B (Anomalib)
│   └── pipeline_c.py           # Pipeline C (RF-DETR)
├── cli/
│   ├── train_cmd.py            # train / predict / benchmark 命令
│   └── predict_cmd.py          # serve / viewer 命令
├── api/                        # RESTful API
│   ├── app.py                  # FastAPI 应用工厂
│   ├── routes.py               # 路由定义
│   ├── schemas.py              # 请求/响应模型
│   └── response_builder.py     # 响应构建
├── viewers/                    # 可视化工具
│   ├── dataset_viewer.py       # Gradio 数据集浏览器
│   ├── dataset_viewer_st.py    # Streamlit 版
│   └── training_workbench.py   # 训练工作台
├── dataset/                    # 数据集（COCO 格式）
└── logs/                       # 按日轮转日志
```

***

## 一、Pipeline 说明

本项目提供三条检测管线：

| Pipeline | 模型                        | 特点               |  当前推荐 |
| :------: | ------------------------- | ---------------- | :---: |
|     A    | YOLOv8                    | 直接目标检测，速度快       |   -   |
|     B    | Anomalib PatchCore + YOLO | 异常检测 + 分类，零样本    |   -   |
|   **C**  | **RF-DETR Medium**        | **端到端目标检测，精度最高** | **✅** |

**当前默认使用 Pipeline C (RF-DETR Medium)**，在 `batch1_coco_p03_aligned_crop2` 数据集上的表现：

| 指标              | 值                 |
| --------------- | ----------------- |
| 精确率 (Precision) | 85.0%             |
| 召回率 (Recall)    | 94.4%             |
| F1 Score        | 89.5%             |
| 漏检率             | 1/18（仅 1 张缺陷图被漏检） |
| 误报率             | 3/60（3 张正常图被误报）   |

***

## 二、模型推理

### 2.1 单图/批量推理

```bash
python main.py predict --pipeline c --variant m --model <模型路径> --source <图片路径或目录> [选项]
```

| 参数             | 默认值     | 说明                       |
| -------------- | ------- | ------------------------ |
| `--pipeline`   | `a`     | Pipeline 类型: a/b/c       |
| `--variant`    | `s`     | Pipeline C 模型变体: n/s/m/l |
| `--model`      | （必填）    | 模型权重文件路径                 |
| `--source`     | （必填）    | 图片路径或目录                  |
| `--conf`       | `0.25`  | 置信度阈值                    |
| `--device`     | `auto`  | 设备                       |
| `--slice-size` | `640`   | 切片大小                     |
| `--overlap`    | `0.25`  | 切片重叠率                    |
| `--no-slice`   | `false` | 禁用切片                     |

**示例**：

```bash
# Pipeline C 单张推理（推荐）
python main.py predict \
  --pipeline c --variant m \
  --model models/pipeline_c_rfdetr_medium_best_ema.pth \
  --source dataset/batch1/异物/S0000093/S0000093_P00_xxx.jpg

# Pipeline C 目录批量推理
python main.py predict \
  --pipeline c --variant m \
  --model models/pipeline_c_rfdetr_medium_best_ema.pth \
  --source dataset/batch1/异物/S0000093
```

**输出示例**：

```
图片: S0000093_P00_xxx.jpg
  耗时: total=2330.0ms
  [defect] conf=0.873 region=(0.1234,0.2345,0.3456,0.4567)
```

### 2.2 推理性能

在 RTX 3080 (10GB) + CUDA 环境下，随机抽取 10 张缺陷图片的推理统计：

| 指标       | 耗时                  |
| -------- | ------------------- |
| **平均耗时** | **4426 ms (4.4 秒)** |
| 中位数      | 4394 ms             |
| 最快       | 4329 ms             |
| 最慢       | 4575 ms             |

> 每张大图（5472×3648）会被切成约 77 片 640×640 的切片逐一推理再合并。

### 2.3 性能基准测试

```bash
python main.py benchmark \
  --model models/pipeline_c_rfdetr_medium_best_ema.pth \
  --source dataset/batch1/异物/S0000093 \
  --runs 10
```

***

## 三、RESTful API 服务

### 3.1 启动服务

```bash
python main.py serve --pipeline c --variant m --model <模型路径> [选项]
```

| 参数           | 默认值       | 说明                 |
| ------------ | --------- | ------------------ |
| `--pipeline` | `a`       | Pipeline 类型: a/b/c |
| `--variant`  | `s`       | Pipeline C 模型变体    |
| `--model`    | （必填）      | 模型权重文件路径           |
| `--host`     | `0.0.0.0` | 监听地址               |
| `--port`     | `8000`    | 监听端口               |
| `--conf`     | `0.25`    | 默认置信度阈值            |
| `--device`   | `auto`    | 推理设备               |
| `--no-slice` | `false`   | 禁用切片推理             |

**示例**：

```bash
python main.py serve \
  --pipeline c --variant m \
  --model models/pipeline_c_rfdetr_medium_best_ema.pth \
  --port 8000 --conf 0.25
```

启动后访问：

- API 文档（Swagger）：`http://localhost:8000/docs`
- OpenAPI Schema：`http://localhost:8000/openapi.json`

### 3.2 API 接口

#### POST /get\_latest\_defect\_infos — 缺陷检测

提交图片列表，返回检测结果。

**请求体**：

```json
{
  "job_id": "JOB20260428001",
  "sample_id": "S0000093",
  "file_names": [
    "S0000093_P00_20260419163509_20260419163457.jpg",
    "S0000093_P03_20260419163522_20260419163457.jpg"
  ],
  "relative_dir": "dataset/batch1"
}
```

| 字段             | 类型        | 必填 | 说明             |
| -------------- | --------- | -- | -------------- |
| `job_id`       | string    | 是  | 任务批次 ID        |
| `sample_id`    | string    | 是  | 样品 ID          |
| `file_names`   | string\[] | 是  | 图片文件名列表，至少 1 个 |
| `relative_dir` | string    | 是  | 图片目录的相对路径      |

**响应体**：

```json
{
  "code": 200,
  "message": "success",
  "job_id": "JOB20260428001",
  "sample_id": "S0000093",
  "position": "P00",
  "relative_dir": "dataset/batch1",
  "timestamp": "1714023926000",
  "defect_infos": [
    {
      "file_name": "S0000093_P00_20260419163509_20260419163457.jpg",
      "defect_list": [
        {
          "type": "defect",
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

| 字段                                                   | 说明                  |
| ---------------------------------------------------- | ------------------- |
| `code`                                               | 200=成功，500=模型未加载    |
| `defect_infos[].defect_list[].type`                  | 缺陷类型名称              |
| `defect_infos[].defect_list[].defect_infos[].region` | 归一化坐标 `x1,y1,x2,y2` |
| `defect_infos[].defect_list[].defect_infos[].conf`   | 置信度 0.0\~1.0        |

#### GET /health — 健康检查

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_version": "pipeline_c_rfdetr_medium_best_ema",
  "classes": ["defect"],
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "pipeline_type": "C"
}
```

#### GET /model/info — 模型信息

```bash
curl http://localhost:8000/model/info
```

```json
{
  "model_path": "models/pipeline_c_rfdetr_medium_best_ema.pth",
  "classes": ["defect"],
  "slicer_enabled": true,
  "slice_size": 640,
  "pipeline_type": "C"
}
```

### 3.3 调用示例

**Python（httpx）**：

```python
import httpx

resp = httpx.post("http://localhost:8000/get_latest_defect_infos", json={
    "job_id": "TEST001",
    "sample_id": "S0001",
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
    "file_names": ["image1.jpg"],
    "relative_dir": "/path/to/images"
  }'
```

### 3.4 日志

- **控制台**：带完整日期时间戳和阶段标签 `[准备]` `[请求]` `[推理]` `[响应]`
- **文件**：按日保存在 `logs/app_YYYYMMDD.log`，默认保留 30 天
- 每个请求分配唯一 ID，贯穿全链路日志

```
2026-04-28 20:23:35 | INFO     | [准备] 模型就绪 | Pipeline C | 类别: ['defect'] | 置信度阈值: 0.25
2026-04-28 20:26:03 | INFO     | [请求] a1b2c3d4 | 收到检测请求 | job_id=TEST001, 文件数=1
2026-04-28 20:26:03 | INFO     | [推理] a1b2c3d4 | [1/1] 开始推理: image1.jpg
2026-04-28 20:26:05 | INFO     | [推理] a1b2c3d4 | [1/1] 完成: 3个缺陷 | total=2330ms
2026-04-28 20:26:05 | INFO     | [响应] a1b2c3d4 | 请求完成 | 总耗时=2552ms
```

***

## 四、Docker 容器化部署

### 4.1 镜像信息

| 项目      | 说明                                                  |
| ------- | --------------------------------------------------- |
| 基础镜像    | `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`     |
| CUDA 版本 | 12.6                                                |
| 内嵌模型    | RF-DETR Medium（defect 检测，已训练好）                      |
| 容器内路径   | `/app/`（工作目录）                                       |
| 模型路径    | `/app/models/pipeline_c_rfdetr_medium_best_ema.pth` |
| 暴露端口    | 8000                                                |
| 默认启动    | Pipeline C (RF-DETR Medium)，无需额外参数                  |

> **模型已打包在镜像内**，启动即可用，无需挂载模型文件。

### 4.2 构建镜像

```powershell
cd f:\Bear\apple
docker build -t b-anomaly:latest .
```

构建过程包括：拉取基础镜像 → 安装系统依赖 → 安装 Python 包 → 拷贝 RF-DETR Medium 模型权重。首次构建约需 10-20 分钟。

### 4.3 快速启动

由于模型已内嵌在镜像中，**只需挂载图片目录即可启动**：

```powershell
docker run -d --gpus all -p 8000:8000 `
  -v D:\production_images:/data/images `
  --name apple-api b-anomaly:latest
```

调用 API 时，`relative_dir` 填容器内的路径：

```json
{
  "job_id": "JOB001",
  "sample_id": "S0001",
  "file_names": ["P00_20260426_001.jpg", "P03_20260426_002.jpg"],
  "relative_dir": "/data/images/JOB001"
}
```

### 4.4 目录挂载说明

容器是隔离环境，API 通过 `relative_dir` 参数定位图片，这个路径是**容器内部的路径**。

核心思路：

1. 容器启动时，用 `-v` 把宿主机的图片目录挂载到容器内的一个固定位置
2. 调用 API 时，`relative_dir` 直接填这个固定位置

**多个目录的情况**：

```powershell
docker run -d --gpus all -p 8000:8000 `
  -v D:\line_a_images:/data/line_a `
  -v E:\line_b_images:/data/line_b `
  --name apple-api b-anomaly:latest
```

```json
{
  "job_id": "JOB001",
  "sample_id": "S0001",
  "file_names": ["P00_20260426_001.jpg"],
  "relative_dir": "/data/line_a/JOB001"
}
```

**开发/测试时使用项目自带的数据集**：

```powershell
docker run -d --gpus all -p 8000:8000 `
  -v f:\Bear\apple\dataset:/app/dataset `
  --name apple-api b-anomaly:latest
```

**总结**：

| 步骤           | 你做什么                         |
| ------------ | ---------------------------- |
| 启动容器         | `-v 宿主机目录:容器内目录`，把图片目录挂进去    |
| 调用 API       | `relative_dir` 填**容器内目录**的路径 |
| `file_names` | 只填文件名，不带路径                   |

### 4.5 验证容器运行

**方法一：查看容器日志**

```powershell
docker logs apple-api
```

正常输出应包含：

- `使用 GPU: NVIDIA GeForce RTX 3080 (10.0 GB)`
- `Pipeline C 初始化完成. 类别: ['defect']`
- `Uvicorn running on http://0.0.0.0:8000`

**方法二：健康检查**

```powershell
python -c "import urllib.request,json; r=urllib.request.urlopen('http://localhost:8000/health'); print(json.dumps(json.loads(r.read()),indent=2,ensure_ascii=False))"
```

正常响应：

```json
{
  "status": "ok",
  "model_version": "pipeline_c_rfdetr_medium_best_ema",
  "classes": ["defect"],
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "pipeline_type": "C"
}
```

**方法三：Swagger 交互式文档**

浏览器打开 `http://localhost:8000/docs`，可以直接在页面上测试所有 API 接口。

**方法四：推理测试**

```powershell
python -c "
import urllib.request, json
payload = json.dumps({
    'job_id': 'test-001',
    'sample_id': 'sample-001',
    'file_names': ['image.jpg'],
    'relative_dir': '/data/images'
}).encode('utf-8')
req = urllib.request.Request('http://localhost:8000/get_latest_defect_infos', data=payload, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
print(f'status: {result[\"code\"]}')
"
```

### 4.6 常用容器管理命令

```powershell
# 查看容器状态
docker ps

# 查看实时日志
docker logs -f apple-api

# 停止容器
docker stop apple-api

# 重新启动
docker start apple-api

# 删除容器
docker rm -f apple-api

# 进入容器调试
docker exec -it apple-api bash
```

### 4.7 性能参考

在 RTX 3080 (10GB) + CUDA 12.6 环境下，Pipeline C (RF-DETR Medium) 的推理性能：

| 场景                          | 耗时           |
| --------------------------- | ------------ |
| 完整高分辨率图（5472×3648，约 77 片切片） | 平均 **4.4 秒** |
| 波动范围                        | 4.3 \~ 4.6 秒 |

***

## 五、模型训练

### 5.1 训练命令

```bash
python main.py train --pipeline <a/b/c> --dataset <数据集路径> [选项]
```

**Pipeline C (RF-DETR) 训练参数**：

| 参数             | 默认值     | 说明                                                      |
| -------------- | ------- | ------------------------------------------------------- |
| `--pipeline`   | `a`     | Pipeline 类型: a(YOLO) / b(Anomalib) / c(RF-DETR)         |
| `--variant`    | `s`     | RF-DETR 模型变体: n(Nano) / s(Small) / m(Medium) / l(Large) |
| `--dataset`    | （必填）    | COCO 格式数据集目录路径                                          |
| `--epochs`     | `100`   | 训练轮数                                                    |
| `--batch-size` | `16`    | 批量大小                                                    |
| `--img-size`   | `640`   | 输入图片尺寸                                                  |
| `--device`     | `auto`  | 设备：auto / cpu / cuda                                    |
| `--output-dir` | `runs`  | 输出目录                                                    |
| `--slice-size` | `640`   | 切片大小（像素）                                                |
| `--overlap`    | `0.25`  | 切片重叠率                                                   |
| `--no-slice`   | `false` | 禁用切片，直接训练                                               |

**Pipeline C 训练示例**：

```bash
python main.py train \
  --pipeline c --variant m \
  --dataset dataset/batch1_coco_p03_aligned_crop2 \
  --epochs 30 \
  --batch-size 4 \
  --device cuda
```

**Pipeline A (YOLO) 训练示例**：

```bash
python main.py train \
  --pipeline a \
  --dataset dataset/batch1_coco \
  --model yolov8n.pt \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

### 5.2 训练结果参考

在 RTX 3080 (10GB) 上，各 Pipeline 的训练指标：

| Pipeline | 模型                 | 切片      | mAP50 / F1     | 训练耗时     |
| -------- | ------------------ | ------- | -------------- | -------- |
| **C**    | **RF-DETR Medium** | **640** | **F1 = 89.5%** | \~2 小时   |
| A        | YOLOv8n            | 640     | mAP50 = 0.586  | \~24 分钟  |
| A        | YOLOv8n            | 无       | mAP50 = 0.511  | \~7.5 分钟 |

> Pipeline C (RF-DETR Medium) 在当前数据集上效果最佳。详细报告见 `docs/training_report_2026-04-28.md`。

***

## 六、其他工具

### 数据集浏览器

```bash
python main.py viewer --port 7860
```

基于 Gradio 的可视化工具，支持浏览数据集图片和标注。

### 配置文件

| 文件                           | 用途                   |
| ---------------------------- | -------------------- |
| `config/default.yaml`        | 全局默认配置（设备、切片、日志、API） |
| `config/pipeline_a.yaml`     | Pipeline A 训练配置      |
| `config/pipeline_b.yaml`     | Pipeline B 训练配置      |
| `config/pipeline_c.yaml`     | Pipeline C 训练配置      |
| `config/defect_mapping.yaml` | 缺陷类别映射与合并规则          |

***

## 技术栈

| 组件     | 技术                           | 说明                        |
| ------ | ---------------------------- | ------------------------- |
| 检测模型   | RF-DETR / YOLOv8 / PatchCore | 三条 Pipeline 可选，默认 RF-DETR |
| 切片推理   | 自研切片引擎                       | 大图自动切片 + NMS 合并           |
| Web 框架 | FastAPI + Uvicorn            | 异步高性能，自带 Swagger 文档       |
| 数据校验   | Pydantic v2                  | 请求/响应/配置模型                |
| 日志     | loguru                       | 按日轮转，带阶段标签                |
| 可视化    | Gradio / Streamlit           | 数据集浏览器 + 训练工作台            |
| 容器化    | Docker + CUDA 12.6           | GPU 直通，镜像内嵌训练好的模型         |

