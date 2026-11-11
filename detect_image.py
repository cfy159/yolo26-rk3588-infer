"""单张图片推理示例。

用法:
    python detect_image.py bus.jpg
    python detect_image.py bus.jpg --show --conf 0.3
    python detect_image.py bus.jpg --onnx-model yolo26s.onnx      # PC 上验证（需 onnxruntime）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from yolo26_rknn.cli import add_detector_args, build_detector
from yolo26_rknn.utils import imread, imwrite
from yolo26_rknn.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 RK3588 单图推理", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("-o", "--output", default=None, help="结果图片保存路径（默认 <原名>_result.jpg）")
    parser.add_argument("--save-txt", action="store_true", help="同时保存 YOLO 格式标签")
    parser.add_argument("--show", action="store_true", help="弹出结果窗口")
    add_detector_args(parser)
    return parser.parse_args()


def default_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}_result{path.suffix or '.jpg'}")


def save_yolo_txt(path: Path, detections, width: int, height: int) -> None:
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {det.confidence:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    args = parse_args()
    image = imread(args.image)
    if image is None:
        print(f"读取图片失败: {args.image}", file=sys.stderr)
        return 1

    detector = build_detector(args)
    try:
        detections = detector.detect(image)
    finally:
        detector.release()

    output_path = Path(args.output) if args.output else default_output(Path(args.image))
    annotated = draw_detections(image, detections)
    if not imwrite(output_path, annotated):
        print(f"保存结果失败: {output_path}", file=sys.stderr)
        return 1

    print(f"\n检测到 {len(detections)} 个目标（conf>={args.conf}）:")
    for det in detections:
        print(f"  {det}")

    timings = detector.last_timings
    print(
        "\n耗时: 预处理 {:.2f} ms | 推理 {:.2f} ms | 后处理 {:.2f} ms | 合计 {:.2f} ms".format(
            timings["preprocess_ms"], timings["inference_ms"], timings["postprocess_ms"], timings["total_ms"]
        )
    )
    print(f"结果图片: {output_path}")

    if args.save_txt:
        txt_path = output_path.with_suffix(".txt")
        height, width = image.shape[:2]
        save_yolo_txt(txt_path, detections, width, height)
        print(f"标签文件: {txt_path}")

    if args.show:
        cv2.imshow("YOLO26 RKNN", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
