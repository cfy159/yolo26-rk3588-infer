"""摄像头 / 视频流推理示例。

用法:
    python detect_camera.py                 # 默认摄像头 0
    python detect_camera.py 2               # USB 摄像头索引 2
    python detect_camera.py test.mp4        # 视频文件
    python detect_camera.py rtsp://ip/...   # RTSP 流
    python detect_camera.py --no-show       # 无显示器时只打印 FPS
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from yolo26_rknn.cli import add_detector_args, build_detector
from yolo26_rknn.utils import is_display_available, parse_video_source
from yolo26_rknn.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO26 RK3588 摄像头/视频推理", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("source", nargs="?", default="0", help="摄像头索引、视频路径或 RTSP 地址")
    parser.add_argument("--width", type=int, default=0, help="摄像头分辨率宽（0 表示不设置）")
    parser.add_argument("--height", type=int, default=0, help="摄像头分辨率高（0 表示不设置）")
    parser.add_argument("--show", dest="show", action="store_true", help="显示实时画面")
    parser.add_argument("--no-show", dest="show", action="store_false", help="不显示画面")
    parser.set_defaults(show=None)
    parser.add_argument("--save-video", default=None, help="保存标注视频路径")
    parser.add_argument("--max-frames", type=int, default=0, help="最多处理帧数（0 表示不限制）")
    parser.add_argument("--print-every", type=int, default=30, help="无窗口时每 N 帧打印一次")
    add_detector_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    show = is_display_available() if args.show is None else args.show

    detector = build_detector(args)
    detector.warmup(3)

    source = parse_video_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        detector.release()
        print(f"无法打开视频源: {args.source!r}（检查设备索引/路径/权限）", file=sys.stderr)
        return 1
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save_video, fourcc, 25.0, (frame_w, frame_h))

    frame_count = 0
    total_dets = 0
    loop_times = deque(maxlen=60)
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if source is not None and isinstance(source, int):
                    print("读取帧失败，摄像头可能已断开", file=sys.stderr)
                break

            loop_start = time.perf_counter()
            detections = detector.detect(frame)
            total_dets += len(detections)
            annotated = draw_detections(frame, detections)
            frame_count += 1

            timings = detector.last_timings
            loop_times.append(time.perf_counter() - loop_start)
            fps = len(loop_times) / max(sum(loop_times), 1e-6)

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.putText(
                    annotated, f"FPS {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow("YOLO26 RKNN", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            elif args.print_every > 0 and frame_count % args.print_every == 0:
                print(
                    f"frame={frame_count} fps={fps:.1f} dets={len(detections)} "
                    f"pre={timings['preprocess_ms']:.1f}ms infer={timings['inference_ms']:.1f}ms "
                    f"post={timings['postprocess_ms']:.1f}ms"
                )

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    finally:
        elapsed = time.perf_counter() - start
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        detector.release()

    avg_fps = frame_count / elapsed if elapsed > 0 else 0.0
    print(f"\n共处理 {frame_count} 帧, 平均 {avg_fps:.2f} FPS, 累计 {total_dets} 个检测框")
    return 0


if __name__ == "__main__":
    sys.exit(main())
