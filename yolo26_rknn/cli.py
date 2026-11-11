"""命令行示例共用的参数与检测器构建逻辑。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .backends import OnnxBackend
from .config import YOLO26Config
from .detector import YOLO26Detector

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_DIR / "yolo26s_rknn_model" / "yolo26s-rk3588.rknn"


def add_detector_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("检测器参数")
    group.add_argument("--model", default=str(DEFAULT_MODEL), help="RKNN 模型路径")
    group.add_argument("--metadata", default=None, help="metadata.yaml 路径（默认取模型同目录）")
    group.add_argument("--imgsz", type=int, default=None, help="模型输入尺寸（默认读 metadata 或 640）")
    group.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    group.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    group.add_argument("--max-det", type=int, default=300, help="最多保留检测框数")
    group.add_argument("--no-nms", action="store_true", help="关闭 NMS（使用 one2one/NMS-free 导出时开启）")
    group.add_argument("--agnostic-nms", action="store_true", help="跨类别 NMS")
    group.add_argument("--core-mask", default="0_1_2", help="NPU 核：0/1/2/0_1/0_1_2")
    group.add_argument("--onnx-model", default=None, help="PC 调试用 ONNX 模型（需要 onnxruntime，替代 RKNN）")
    group.add_argument("--quiet", action="store_true", help="不打印模型加载信息")
    return parser


def build_detector(args: argparse.Namespace) -> YOLO26Detector:
    backend: Optional[OnnxBackend] = None
    model_path = args.model
    if args.onnx_model:
        model_path = args.onnx_model
        backend = OnnxBackend(Path(args.onnx_model))

    config = YOLO26Config(
        model_path=model_path,
        metadata_path=args.metadata,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        use_nms=not args.no_nms,
        class_agnostic_nms=args.agnostic_nms,
        core_mask=args.core_mask,
        verbose=not args.quiet,
    )
    return YOLO26Detector(config, backend=backend)
