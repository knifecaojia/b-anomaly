from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from core.config import APILogConfig


def setup_logging(log_config: APILogConfig | None = None) -> None:
    cfg = log_config or APILogConfig()
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=cfg.level.upper(),
    )
    log_dir = Path(cfg.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / "app_{time:YYYYMMDD}.log")
    logger.add(
        log_file,
        rotation="1 day",
        retention=f"{cfg.retention_days} days",
        encoding="utf-8",
        level="DEBUG",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="工业产品缺陷检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    from cli.train_cmd import (
        create_benchmark_parser,
        create_predict_parser,
        create_train_parser,
        run_benchmark,
        run_predict,
        run_train,
    )
    from cli.predict_cmd import (
        create_serve_parser,
        create_viewer_parser,
        run_serve,
        run_viewer,
    )

    create_train_parser(subparsers)
    create_predict_parser(subparsers)
    create_benchmark_parser(subparsers)
    create_serve_parser(subparsers)
    create_viewer_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging()

    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "viewer":
        run_viewer(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
