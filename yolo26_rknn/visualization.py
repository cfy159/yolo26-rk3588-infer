"""检测结果可视化。"""

from __future__ import annotations

import colorsys
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .results import Detection


def class_color(class_id: int) -> Tuple[int, int, int]:
    """用黄金比例分散色相，保证相邻类别颜色差异明显，颜色按类别固定。"""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR


def draw_detections(
    image_bgr: np.ndarray,
    detections: Sequence[Detection],
    names: Optional[List[str]] = None,
    show_confidence: bool = True,
    thickness: Optional[int] = None,
    copy: bool = True,
) -> np.ndarray:
    """在图上画检测框，返回标注后的图像（默认不修改原图）。"""
    canvas = image_bgr.copy() if copy else image_bgr
    height, width = canvas.shape[:2]
    if thickness is None:
        thickness = max(1, int(round(min(height, width) / 400)))
    font_scale = max(0.4, min(height, width) / 1100.0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for det in detections:
        color = class_color(det.class_id)
        x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        name = names[det.class_id] if names and 0 <= det.class_id < len(names) else det.class_name
        label = f"{name} {det.confidence:.2f}" if show_confidence else str(name)
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(y1 - text_h - baseline - 2, 0)
        label_left = max(0, min(x1, width - text_w - 4))
        cv2.rectangle(
            canvas, (label_left, label_top), (label_left + text_w + 2, label_top + text_h + baseline + 2), color, -1
        )
        cv2.putText(
            canvas, label, (label_left + 1, label_top + text_h + 1), font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )
    return canvas
