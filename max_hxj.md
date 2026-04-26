# 工业产品缺陷检测系统 — 项目说明

## 项目是什么

想象一个工厂流水线上的质检员，他需要盯着每一件产品，看上面有没有划痕、异物、过切等缺陷。这个项目就是用 AI 来替代这个质检员。

系统能做两件事：
1. **学习**：你给它一批标注好的缺陷图片，它自己学会识别
2. **检测**：学会之后，你传一张图片给它，它告诉你图片上哪里有缺陷、是什么缺陷、有多大把握

## 为什么这样设计

### 1. 切片策略：把大图切成小块

工厂拍的照片分辨率是 5472×3648（约 2000 万像素），但 AI 模型只能处理 640×640 的图片。

**最简单的方案**是把大图缩到 640×640，但这样做就像把一张高清照片压成缩略图，缺陷的细节全丢了。

**我们的方案**是把大图切成约 77 个 640×640 的小块（每块和相邻块有 25% 的重叠，防止缺陷正好在切割线上被截断），让 AI 对每个小块分别检测，最后把所有小块的结果合并回去。

这就像你拿着放大镜，把整张大图逐块看过去，不会遗漏任何细节。

### 2. 双路线计划

我们计划了两条检测路线：

- **路线A（已实现）**：直接训练一个 YOLO 模型，一步到位地识别缺陷类型和位置
- **路线B（计划中）**：先用异常检测模型找出"不正常"的区域，再用分类器判断具体是什么缺陷

路线B的优势是只需要"正常产品"的图片就能训练，不需要标注缺陷。这对新产线特别有用——刚上线时你可能还没有缺陷样本。

### 3. 为什么用 YOLO

YOLO 是目前最主流的目标检测模型之一，它的特点是快。对于工业检测这种需要实时响应的场景，YOLO 是最务实的选择。我们用的是 Ultralytics 的 YOLOv8 实现，开箱即用，训练和推理都很方便。

## 项目结构

```
apple/
├── config/          # 配置文件（训练参数、缺陷类别映射）
├── core/            # 核心逻辑
│   ├── config.py    # 配置加载和校验（启动时就报错，不在运行时才崩）
│   ├── device.py    # 自动检测 GPU/CPU
│   ├── timing.py    # 计时器（记录每一步的耗时）
│   ├── slicer.py    # 切片处理器（大图→小块，小块→合并结果）
│   ├── dataset_manager.py  # 数据集管理（格式转换、统计）
│   └── yolo_engine.py      # YOLO 引擎（训练+推理的核心）
├── pipeline/        # 流水线层（编排整个检测流程）
├── api/             # Web API（让其他系统调用检测服务）
├── cli/             # 命令行工具（直接在终端使用）
├── viewers/         # 可视化界面（Gradio，浏览器里看数据和训练进度）
├── main.py          # 统一入口
└── dataset/         # 数据集（训练用的图片和标注）
```

## 怎么用

### 训练模型

```bash
python main.py train --dataset dataset/batch1_coco --epochs 100 --batch-size 8
```

### 命令行推理

```bash
python main.py predict --model runs/train_xxx/weights/best.pt --source test.jpg
```

### 启动 API 服务

```bash
python main.py serve --model runs/train_xxx/weights/best.pt --port 8000
```

然后其他系统就可以通过 HTTP POST 请求来调用检测服务了。

### 打开可视化界面

```bash
python main.py viewer
```

在浏览器里就能看到数据集、切片预览、训练进度。

## 数据情况

当前可用的标注数据：
- **异物**：246 个标注（有东西不该出现在产品上）
- **过切**（小NC过切 + 过切合并）：305 个标注（加工过头了）

还有 695 个破损类标注分布在 18 个目录中，需要清洗确认后才能使用。

## 踩过的坑和经验

### 1. PowerShell 不认 &&

在 PowerShell 里，`&&` 不是有效的命令连接符。用分号 `;` 代替。

### 2. 高分辨率图片不能直接推理

第一次尝试直接把 5472×3648 的图片送入模型，结果要么显存爆了（OOM），要么缩放到 640×640 后缺陷特征全无。切片策略是必须的。

### 3. 接口文档中的类别名只是示例

接口文档里写的 bent/fiber/scratch 只是举例说明格式，实际类别必须从数据集标注中获取。硬编码这些类别会导致后续无法扩展。

### 4. 负样本很重要

每张高分辨率图切成约 77 个小块，其中只有 3-5 个含缺陷。如果不加负样本（无缺陷的背景块），模型会倾向于把所有东西都判为缺陷。按正负比 1:3~1:5 采样背景块，能有效降低误检率。

### 5. Docker 镜像选择：runtime 而非 devel

PyTorch 官方提供两种镜像：`devel`（包含完整编译工具链，约 7-8GB）和 `runtime`（只含运行时，约 3-4GB）。对于推理部署场景，runtime 就够了。devel 镜像体积太大，下载过程中 Docker 引擎容易超时崩溃。

关键经验：如果依赖包安装时需要编译（比如某些 C 扩展），在 runtime 镜像基础上通过 `apt-get install gcc g++` 补装编译器即可，没必要用整个 devel 镜像。

### 6. 容器化部署时目录挂载是刚需

API 的核心接口接收 `relative_dir` 参数来读取图片。如果图片目录没有挂载到容器内，API 就找不到文件。所以 `docker run` 时必须用 `-v` 把本地的 dataset 目录挂载进去。

### 7. 代码中隐藏的 bug 会在容器测试时暴露

在容器测试推理接口时，发现 `routes.py` 中引用了 `request.position`，但 `DefectRequest` 模型根本没有 `position` 字段。这个 bug 之前没被触发，可能是因为之前测试走的路径不经过这段代码。教训：**每个分支都要测试到，尤其是错误处理分支**。

## Docker 部署

项目支持 Docker 容器化部署，以下是关键信息：

**Dockerfile**：基于 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`，内含 yolov8n.pt 模型。

**构建镜像**：
```powershell
docker build -t b-anomaly:latest .
```

**运行容器**：
```powershell
docker run -d --gpus all -p 8000:8000 `
  -v f:\Bear\apple\dataset:/app/dataset `
  -v f:\Bear\apple\config:/app/config `
  --name apple-api b-anomaly:latest
```

**测试验证**：
- 健康检查：`GET http://localhost:8000/health`（确认 device=cuda）
- 模型信息：`GET http://localhost:8000/model/info`
- 推理测试：`POST http://localhost:8000/get_latest_defect_infos`

## 技术栈

| 用途 | 技术 |
|------|------|
| 目标检测 | Ultralytics YOLOv8 |
| 切片处理 | sahi + 自研 |
| Web 服务 | FastAPI + Uvicorn |
| 可视化 | Gradio |
| 配置校验 | Pydantic |
| 日志 | loguru |
| 数据标注 | supervision |
| 容器化 | Docker (CUDA 12.6 + PyTorch 2.6.0) |
