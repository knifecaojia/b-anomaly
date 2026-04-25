from __future__ import annotations

import argparse


def create_serve_parser(subparsers) -> None:
    parser = subparsers.add_parser("serve", help="启动 RESTful API 服务")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--conf", type=float, default=0.25, help="默认置信度阈值")
    parser.add_argument("--device", type=str, default="auto", help="设备")
    parser.add_argument("--no-slice", action="store_true", help="禁用切片推理")


def run_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from api.app import create_app

    app = create_app(
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device,
        disable_slicing=args.no_slice,
    )

    uvicorn.run(app, host=args.host, port=args.port)


def create_viewer_parser(subparsers) -> None:
    parser = subparsers.add_parser("viewer", help="启动 Gradio 综合工作台")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    parser.add_argument("--port", type=int, default=7860, help="端口")


def run_viewer(args: argparse.Namespace) -> None:
    from viewers.dataset_viewer import create_viewer_app

    app = create_viewer_app()
    app.launch(share=args.share, server_port=args.port)
