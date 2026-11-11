"""与推理无关的小工具：兼容中文路径的读写、视频源解析。"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


def imread(path: Union[str, Path]) -> Optional[np.ndarray]:
    """读取图片；Windows 下 cv2.imread 不支持中文路径，用 imdecode 兜底。"""
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR) if path.isascii() else None
    if image is not None:
        return image
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, cv2.error):
        return None


def imwrite(path: Union[str, Path], image: np.ndarray) -> bool:
    """保存图片；同样兼容中文路径。"""
    path = str(path)
    suffix = Path(path).suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        encoded.tofile(path)
        return True
    except OSError:
        return False


def is_display_available() -> bool:
    """判断当前环境能否创建 OpenCV 窗口（板端常为 headless）。"""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def parse_video_source(source: str) -> Union[int, str]:
    """'0' -> 摄像头索引 0；其余按路径 / RTSP 地址处理。"""
    text = str(source)
    return int(text) if text.isdigit() else text
