"""YOLO26 官方导出格式的后处理。

模型输出（RKNN/ONNX 一致）::

    output0: (1, 84, 8400) float
        [0:4]   -> cx, cy, w, h        # 已解码到 640×640 输入尺度（像素）
        [4:84]  -> 80 类分数           # 模型内已做过 sigmoid

类别分数取每行最大值作为该 anchor 的类别与置信度，再按阈值过滤、NMS。

注意：Ultralytics 默认导出的是 one2many 预测，同一目标会有大量重叠框，
因此 use_nms=True。如果后续换成 one2one（NMS-free）头导出的模型，
可以把 use_nms 设为 False。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _to_prediction_matrix(output: np.ndarray, num_classes: Optional[int]) -> np.ndarray:
    """把 (1, 84, 8400) 或 (1, 8400, 84) 统一成 (num_anchors, 4 + num_classes)。"""
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"仅支持 batch=1，当前输出 shape={pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"不支持的输出维度 shape={pred.shape}")

    expected = None if num_classes is None else num_classes + 4
    if expected is not None and pred.shape[0] == expected:
        pred = pred.T
    elif expected is not None and pred.shape[1] == expected:
        pass
    elif pred.shape[0] <= pred.shape[1]:
        # 典型布局 (84, 8400)，通道数远小于 anchor 数
        pred = pred.T

    if expected is not None and pred.shape[1] != expected:
        raise ValueError(
            f"输出通道数 {pred.shape[1]} 与类别数+4={expected} 不匹配。"
            "请检查该 .rknn 是否为单输出 output0 格式，或类别名是否配置正确。"
        )
    return pred


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    """标准 NMS，返回保留下来的下标（按分数从高到低）。"""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)

    boxes = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    order = np.argsort(-scores, kind="stable")
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        inter_w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        inter_h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = inter_w * inter_h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return np.asarray(keep, dtype=np.int64)


_NMS_MAX_WH = 7680.0


def _class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thres: float
) -> np.ndarray:
    """按类别分组做 NMS：给不同类别加坐标偏移，一次调用即可。

    偏移量取 7680（与 Ultralytics 一致），保证不同类别的框在坐标上不会重叠。
    """
    offset = (class_ids.astype(np.float32) * _NMS_MAX_WH)[:, None]
    shifted = boxes + offset
    return nms(shifted, scores, iou_thres)


def decode_predictions(
    outputs: Sequence[np.ndarray],
    conf_thres: float,
    iou_thres: float,
    max_det: int = 300,
    use_nms: bool = True,
    class_agnostic_nms: bool = False,
    num_classes: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """解码 output0，返回 (boxes_xyxy, scores, class_ids)，坐标为模型输入尺度。

    参数:
        outputs: 后端返回的输出列表，当前模型只有 output0 一个元素。
        num_classes: 类别数量（一般等于 names 的长度）。
    """
    if not outputs:
        raise RuntimeError("模型没有返回任何输出")

    pred = _to_prediction_matrix(outputs[0], num_classes)
    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    if class_scores.shape[1] == 0:
        raise ValueError("输出中没有类别分数通道")

    class_ids = class_scores.argmax(axis=1)
    confs = class_scores[np.arange(class_ids.shape[0]), class_ids]

    mask = confs >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    if boxes_xywh.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    cx, cy, bw, bh = boxes_xywh.T
    boxes_xyxy = np.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), axis=1).astype(np.float32)

    # 先按分数取前 max_det 个候选，再做 NMS（与 Ultralytics 顺序一致）
    order = np.argsort(-confs, kind="stable")[:max_det]
    boxes_xyxy = boxes_xyxy[order]
    confs = confs[order]
    class_ids = class_ids[order]

    if use_nms and boxes_xyxy.shape[0] > 0:
        if class_agnostic_nms:
            keep = nms(boxes_xyxy, confs, iou_thres)
        else:
            keep = _class_aware_nms(boxes_xyxy, confs, class_ids, iou_thres)
        boxes_xyxy = boxes_xyxy[keep]
        confs = confs[keep]
        class_ids = class_ids[keep]

    # 保险起见再按 max_det 截断
    boxes_xyxy = boxes_xyxy[:max_det]
    confs = confs[:max_det].astype(np.float32)
    class_ids = class_ids[:max_det].astype(np.int32)
    return boxes_xyxy, confs, class_ids
