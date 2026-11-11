"""YOLO26 检测器：编排预处理 -> 后端推理 -> 后处理。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .backends import Backend, RKNNLiteBackend
from .config import COCO_NAMES, YOLO26Config
from .metadata import ModelMetadata, load_metadata, names_list
from .postprocessing import decode_predictions
from .preprocessing import letterbox, scale_boxes
from .results import Detection


class YOLO26Detector:
    """面向 RK3588 的 YOLO26 检测器。

    输入统一为 OpenCV 的 BGR 图像 (H, W, 3) uint8；
    内部自动完成 letterbox、RGB 转换、NPU 推理、解码、NMS、坐标还原。
    """

    def __init__(self, config: Optional[YOLO26Config] = None, backend: Optional[Backend] = None):
        self.config = config or YOLO26Config()
        self.metadata: Optional[ModelMetadata] = None
        self.last_timings: Dict[str, float] = {}
        self._closed = False

        # 元数据优先从后端实际模型所在目录读取（方便 --onnx-model 调试）
        metadata_path = self.config.metadata_path
        if metadata_path is None:
            backend_path = self.backend_model_path(backend)
            base_dir = backend_path.parent if backend_path is not None else Path(self.config.model_path).parent
            metadata_path = base_dir / "metadata.yaml"
        if metadata_path is not None:
            self.metadata = load_metadata(metadata_path)

        self.names = self._resolve_names()
        self.num_classes = len(self.names)
        self.imgsz = self._resolve_imgsz()

        if self.metadata and self.metadata.task and self.metadata.task != "detect":
            raise ValueError(f"当前推理库只支持 detect 任务，metadata 中 task={self.metadata.task!r}")

        self.backend: Backend = backend or RKNNLiteBackend(
            self.config.model_path, core_mask=self.config.core_mask, verbose=self.config.verbose
        )
        if self.config.verbose:
            print(f"[YOLO26] model={self.config.model_path}")
            print(f"[YOLO26] imgsz={self.imgsz}, classes={self.num_classes}, "
                  f"conf={self.config.conf_thres}, iou={self.config.iou_thres}, nms={self.config.use_nms}")
        self.backend.load()

    @staticmethod
    def backend_model_path(backend: Backend) -> Optional[Path]:
        path = getattr(backend, "model_path", None)
        return Path(path) if path is not None else None

    def _resolve_names(self) -> List[str]:
        if self.config.names is not None:
            return [str(name) for name in self.config.names]
        if self.metadata and self.metadata.names:
            return names_list(self.metadata.names)
        return list(COCO_NAMES)

    def _resolve_imgsz(self) -> int:
        if self.config.imgsz is not None:
            return int(self.config.imgsz)
        if self.metadata and self.metadata.imgsz:
            height, width = self.metadata.imgsz
            if height != width:
                print(f"[YOLO26] 警告: metadata imgsz={height}x{width} 非正方形，本库按 {max(height, width)} 处理")
            return int(max(height, width))
        return 640

    def class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.names):
            return self.names[class_id]
        return str(class_id)

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """对一张 BGR 图像做检测，返回原图坐标系下的检测框。"""
        if self._closed:
            raise RuntimeError("检测器已释放，不能再调用 detect()")
        if image_bgr is None or image_bgr.size == 0:
            return []

        t0 = time.perf_counter()
        lb = letterbox(image_bgr, self.imgsz, pad_value=self.config.pad_value)
        model_input = self.backend.to_model_input(lb.image_rgb)

        t1 = time.perf_counter()
        outputs = self.backend.infer(model_input)

        t2 = time.perf_counter()
        boxes, confs, class_ids = decode_predictions(
            outputs,
            conf_thres=self.config.conf_thres,
            iou_thres=self.config.iou_thres,
            max_det=self.config.max_det,
            use_nms=self.config.use_nms,
            class_agnostic_nms=self.config.class_agnostic_nms,
            num_classes=self.num_classes,
        )
        boxes = scale_boxes(boxes, lb) if boxes.shape[0] else boxes

        detections = [
            Detection(
                xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_name(int(cls_id)),
            )
            for box, conf, cls_id in zip(boxes, confs, class_ids)
        ]
        t3 = time.perf_counter()

        self.last_timings = {
            "preprocess_ms": (t1 - t0) * 1000,
            "inference_ms": (t2 - t1) * 1000,
            "postprocess_ms": (t3 - t2) * 1000,
            "total_ms": (t3 - t0) * 1000,
        }
        return detections

    def warmup(self, runs: int = 3) -> None:
        """用空白图预热 NPU，避免第一次推理耗时偏大影响测速。"""
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(max(1, runs)):
            self.detect(dummy)

    def release(self) -> None:
        if not self._closed:
            self.backend.release()
            self._closed = True

    close = release

    def __enter__(self) -> "YOLO26Detector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
