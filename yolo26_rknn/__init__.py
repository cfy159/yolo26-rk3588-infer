"""YOLO26 RK3588 推理库。

设计目标是模块化：
    YOLO26Detector  只负责流程编排
    backend         只负责模型推理（RKNNLite / ONNX 可互换）
    preprocessing   只负责 letterbox + RGB + 输入组装
    postprocessing  只负责 output0 解码 + NMS
    visualization   只负责画框

典型用法::

    from yolo26_rknn import YOLO26Config, YOLO26Detector
    from yolo26_rknn.utils import imread

    detector = YOLO26Detector(YOLO26Config(model_path="yolo26s_rknn_model/yolo26s-rk3588.rknn"))
    image = imread("bus.jpg")
    detections = detector.detect(image)
    for det in detections:
        print(det.class_name, det.confidence, det.xyxy)
    detector.release()
"""

from __future__ import annotations

from .backends import Backend, OnnxBackend, RKNNLiteBackend, create_backend
from .config import COCO_NAMES, YOLO26Config
from .detector import YOLO26Detector
from .results import Detection
from .visualization import draw_detections

__all__ = [
    "Backend",
    "COCO_NAMES",
    "Detection",
    "OnnxBackend",
    "RKNNLiteBackend",
    "YOLO26Config",
    "YOLO26Detector",
    "create_backend",
    "draw_detections",
]
