# Pipeline B (异常检测) vs Pipeline C (RF-DETR) 对比实验报告

**实验日期**：2026-04-28  
**数据集**：`batch1_coco_p03_aligned_crop2`（SIFT 对齐 + 裁剪黑边后的 P03 工位数据）

---

## 1. 实验目的

对比 Pipeline B 的两种异常检测算法（**PatchCore** 和 **EfficientAD**）与 Pipeline C 的 **RF-DETR** 目标检测在相同数据集下的缺陷检测效果。

---

## 2. 实验配置

| 配置项 | PatchCore | EfficientAD | RF-DETR Medium |
|:---|:---|:---|:---|
| **模型架构** | PatchCore (wide_resnet50_2) | EfficientAD Small | RF-DETR Medium |
| **学习范式** | 仅正常品（无监督） | 仅正常品（无监督） | 有标注框（有监督） |
| **训练数据** | 224 张正常品 | 224 张正常品→12096 切片 | 268 张（59 个缺陷标注）→ 切片 |
| **测试集** | 93 正常 + 67 缺陷 | 93 正常 + 67 缺陷 | 60 正常 + 18 缺陷 |
| **推理方式** | 切片推理（640×640） | 切片推理（640×640） | 切片推理（640×640） |
| **训练输入** | 256×256 全图 | 256×256 切片 | 640×640 切片 |
| **训练耗时** | ~31 秒（1 epoch） | ~4.5 小时（30 epochs） | ~1 小时 |

---

## 3. 测试结果对比

### 核心指标

| 指标 | PatchCore | EfficientAD | RF-DETR Medium |
|:---|:---:|:---:|:---:|
| **精确率 (Precision)** | 44.6% | 41.9% | **85.0%** |
| **召回率 (Recall)** | 98.5% | 100.0% | 94.4% |
| **F1 Score** | 61.4% | 59.0% | **89.5%** |
| **AUROC** | 0.6362 | 0.5712 | - |
| **分离指数 (D)** | 0.261 | -0.025 | - |

### PatchCore 详细分析

- **正常品得分**：均值=100.211，标准差=1.049，最大值=103.003
- **缺陷品得分**：均值=100.802，标准差=1.218，最小值=98.338
- 分离指数 0.261（正值，但很弱）
- 仅 2/67（3.0%）的缺陷品得分超过正常品最高分
- PatchCore 比 EfficientAD 稍好（分离指数从 -0.025 升到 0.261），但仍然远远不够

### EfficientAD 详细分析

- **正常品得分**：均值=4.838，标准差=5.177，最大值=41.521
- **缺陷品得分**：均值=4.628，标准差=3.220，最小值=2.345
- 分离指数为负（-0.025），完全失效
- 0/67（0.0%）的缺陷品超过正常品最高分

---

## 4. 结论

### 三种方法排名

1. **RF-DETR Medium — F1=89.5%** — 明显胜出
2. **PatchCore — F1=61.4%** — 比 EfficientAD 稍好，但远不如 RF-DETR
3. **EfficientAD — F1=59.0%** — 基本失效

### 失败原因分析（PatchCore 和 EfficientAD 共性）

两种异常检测方法都失败了，根本原因相同：

1. **缺陷太小**：异物平均面积仅 1145 像素（占图片的 0.007%），异常信号极弱。
2. **正常品不够一致**：金属/石墨零部件表面本身有丰富的自然纹理变化，模型无法构建紧致的"正常边界"。
3. **方法论局限**：无监督异常检测的前提是"正常品高度一致、缺陷会带来明显像素异常"。我们的场景恰好违反了这个前提。

PatchCore 比 EfficientAD 稍好（AUROC 0.64 vs 0.57），因为 PatchCore 使用预训练的特征提取器（ImageNet 上的 wide_resnet50_2），提取的特征比 EfficientAD 从头学习的特征更有判别力。

### 最终建议

- **RF-DETR（Pipeline C）是唯一可用的选择。** F1 89.5% 远超两个异常检测方法。
- 异常检测方法（PatchCore/EfficientAD）不适合"微小缺陷 + 复杂纹理"的检测场景。
- 如果未来需要检测"从未见过的新型缺陷"，异常检测可作辅助，但不能作为主检测管线。

---

## 附录：实验文件

| 文件 | 说明 |
|:---|:---|
| `convert_coco_to_anomalib.py` | COCO → anomalib 格式转换脚本 |
| `run_efficientad_crop2.py` | EfficientAD 训练+评估脚本 |
| `run_patchcore_crop2.py` | PatchCore 训练+评估脚本 |
| `runs/efficientad_crop2_sliced/eval_results.json` | EfficientAD 评估结果 |
| `runs/patchcore_crop2_sliced/eval_results.json` | PatchCore 评估结果 |
| `dataset/anomalib_crop2/` | 转换后的 anomalib 格式数据集 |
