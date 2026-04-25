本项目是一个以python 为基础的工业产品缺陷检测web服务器

需要具备训练和推理2大任务模式

训练主要在本地通过命令行进行

推理主要提供命令行和restful 服务2种，其中web服务要符合 缺陷识别接口文档.pdf 的要求

需要在logs中保存推理日志

需要分阶段跟踪取图、推理等主要步骤的耗时

需要支持cuda 若无cuda则退回到cpu

尽量使用  ultralytics，roboflow 的 spuervision 等主流库  如果需要缺陷检测的专门的库可以使用 anomalib (https://github.com/open-edge-platform/anomalib)

数据集在 dataset目录下

需要支持一个gradio或者类似的工具来实时查看数据集（coco 或者 yolo格式）

需要支持一个gui 实时查看训练日志 特别是模型的学习曲线
