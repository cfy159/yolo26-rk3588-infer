"""性能测试：连续推理统计预处理 / NPU 推理 / 后处理耗时。

用法:
    python benchmark.py bus.jpg --runs 100
    python benchmark.py bus.jpg --runs 50 --json
    python benchmark.py bus.jpg --onnx-model yolo26s.onnx    # PC 上先验证流程
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from yolo26_rknn.cli import add_detector_args, build_detector
from yolo26_rknn.utils import imread

METRICS = ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 RK3588 性能测试", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("image", nargs="?", default="bus.jpg", help="测试图片")
    parser.add_argument("--runs", type=int, default=100, help="统计次数")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数（不计入统计）")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出")
    add_detector_args(parser)
    return parser.parse_args()


def summarize(values):
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": sorted(values)[max(0, int(len(values) * 0.95) - 1)],
    }


def main() -> int:
    args = parse_args()
    image = imread(args.image)
    if image is None:
        print(f"读取图片失败: {args.image}", file=sys.stderr)
        return 1
    if args.runs <= 0:
        print("--runs 必须大于 0", file=sys.stderr)
        return 1

    detector = build_detector(args)
    try:
        for _ in range(max(0, args.warmup)):
            detector.detect(image)

        records = {key: [] for key in METRICS}
        num_detections = 0
        for _ in range(args.runs):
            detections = detector.detect(image)
            num_detections = len(detections)
            for key in METRICS:
                records[key].append(detector.last_timings[key])
    finally:
        detector.release()

    stats = {key: summarize(values) for key, values in records.items()}
    fps = 1000.0 / stats["total_ms"]["mean"] if stats["total_ms"]["mean"] > 0 else 0.0

    if args.json:
        print(json.dumps(
            {
                "image": args.image,
                "runs": args.runs,
                "detections": num_detections,
                "fps": round(fps, 2),
                "timings_ms": {key: {k: round(v, 3) for k, v in s.items()} for key, s in stats.items()},
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    print(f"\n图片: {args.image}  检测数: {num_detections}  次数: {args.runs}  预热: {args.warmup}")
    print(f"{'阶段':<16}{'mean':>10}{'median':>10}{'p95':>10}{'min':>10}{'max':>10}")
    labels = {
        "preprocess_ms": "预处理",
        "inference_ms": "NPU/模型推理",
        "postprocess_ms": "后处理",
        "total_ms": "合计",
    }
    for key in METRICS:
        s = stats[key]
        print(f"{labels[key]:<14}{s['mean']:>10.2f}{s['median']:>10.2f}{s['p95']:>10.2f}{s['min']:>10.2f}{s['max']:>10.2f}")
    print(f"\n平均 FPS（单帧串行）: {fps:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
