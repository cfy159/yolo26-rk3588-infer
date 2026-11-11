"""推理配置与默认类别名。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

# COCO 80 类，metadata.yaml 缺失时使用
COCO_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)


@dataclass
class YOLO26Config:
    """YOLO26 RKNN 推理配置。

    Attributes:
        model_path: .rknn 模型路径。
        metadata_path: metadata.yaml 路径；None 表示自动在模型同目录查找。
        imgsz: 模型输入尺寸，None 表示从 metadata.yaml 读取，再回退到 640。
        conf_thres: 置信度阈值（类别分数已由模型做过 sigmoid）。
        iou_thres: NMS IoU 阈值。
        max_det: 每张图最多保留的检测框数量。
        use_nms: 是否执行 NMS。官方 Ultralytics 导出的是 one2many 输出，需要 NMS；
            如果模型是按参考项目改成 one2one 头导出的，可以设为 False。
        class_agnostic_nms: 是否跨类别做 NMS（默认按类别分别做）。
        pad_value: letterbox 填充值（Ultralytics 默认 114 灰色）。
        core_mask: RK3588 NPU 核绑定，可选 "0"/"1"/"2"/"0_1"/"0_1_2"，默认三核。
        verbose: 是否打印模型加载信息。
        names: 自定义类别名；None 时优先读 metadata.yaml，再回退 COCO_NAMES。
    """

    model_path: Union[str, Path] = "yolo26s_rknn_model/yolo26s-rk3588.rknn"
    metadata_path: Optional[Union[str, Path]] = None
    imgsz: Optional[int] = None
    conf_thres: float = 0.25
    iou_thres: float = 0.7
    max_det: int = 300
    use_nms: bool = True
    class_agnostic_nms: bool = False
    pad_value: int = 114
    core_mask: Union[str, int] = "0_1_2"
    verbose: bool = True
    names: Optional[Sequence[str]] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.conf_thres <= 1.0:
            raise ValueError(f"conf_thres 必须在 [0, 1] 内，当前 {self.conf_thres}")
        if not 0.0 <= self.iou_thres <= 1.0:
            raise ValueError(f"iou_thres 必须在 [0, 1] 内，当前 {self.iou_thres}")
        if self.max_det <= 0:
            raise ValueError(f"max_det 必须为正整数，当前 {self.max_det}")
        if self.imgsz is not None and self.imgsz <= 0:
            raise ValueError(f"imgsz 必须为正整数，当前 {self.imgsz}")
