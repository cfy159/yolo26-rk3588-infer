"""图像预处理：letterbox（保持宽高比 + 灰色填充）与坐标还原。

必须使用 letterbox 而不是直接 resize，否则框位置会因比例失真而偏移。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterBoxResult:
    """letterbox 结果及坐标还原所需参数。"""

    image_rgb: np.ndarray  # (imgsz, imgsz, 3), uint8, RGB
    ratio: float           # 原图 -> 模型输入的缩放比例
    pad_x: int             # 左侧填充宽度
    pad_y: int             # 顶部填充高度
    orig_w: int
    orig_h: int


def letterbox(
    image_bgr: np.ndarray,
    imgsz: int,
    pad_value: int = 114,
    scaleup: bool = True,
) -> LetterBoxResult:
    """把 BGR 图按比例缩放到 imgsz×imgsz 并居中填充，再转成 RGB。

    Args:
        image_bgr: OpenCV 读入的 BGR 图像 (H, W, 3)。
        imgsz: 模型输入边长（当前模型为 640）。
        pad_value: 填充灰度值，Ultralytics 默认为 114。
        scaleup: 小图是否允许放大（与 Ultralytics 推理默认一致）。
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"期望 HWC 三通道 BGR 图像，实际 shape={image_bgr.shape}")

    height, width = image_bgr.shape[:2]
    ratio = min(imgsz / width, imgsz / height)
    if not scaleup:
        ratio = min(ratio, 1.0)

    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    new_w, new_h = max(new_w, 1), max(new_h, 1)

    # 与 Ultralytics LetterBox 的居中取整方式保持一致
    dw = (imgsz - new_w) / 2
    dh = (imgsz - new_h) / 2
    left = int(round(dw - 0.1))
    top = int(round(dh - 0.1))
    right = int(round(dw + 0.1))
    bottom = int(round(dh + 0.1))

    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h + top + bottom, new_w + left + right, 3), pad_value, dtype=np.uint8)
    canvas[top : top + new_h, left : left + new_w] = resized

    # 模型训练/导出用的是 RGB（RKNN config rgb2bgr=False）
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return LetterBoxResult(canvas_rgb, ratio, left, top, width, height)


def scale_boxes(boxes_xyxy: np.ndarray, lb: LetterBoxResult) -> np.ndarray:
    """把模型输入尺度(640)下的 xyxy 还原到原图坐标并裁剪到图像边界。"""
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)

    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - lb.pad_x) / lb.ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - lb.pad_y) / lb.ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, lb.orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, lb.orig_h)
    return boxes
