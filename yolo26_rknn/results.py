"""推理结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class Detection:
    """单个检测框。

    坐标均为原图坐标系（输入到 detect() 的那张图的像素坐标）。
    """

    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = ""

    @property
    def width(self) -> float:
        return max(0.0, self.xyxy[2] - self.xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.xyxy[3] - self.xyxy[1])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2, (self.xyxy[1] + self.xyxy[3]) / 2)

    def to_dict(self) -> Dict[str, object]:
        return {
            "xyxy": [round(v, 2) for v in self.xyxy],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
        }

    def __str__(self) -> str:
        x1, y1, x2, y2 = (round(v, 1) for v in self.xyxy)
        return f"{self.class_name}({self.class_id}) {self.confidence:.3f} [{x1}, {y1}, {x2}, {y2}]"
