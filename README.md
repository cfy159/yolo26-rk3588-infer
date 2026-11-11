# YOLO26s RK3588 推理（rknnlite）

基于 RKNN-Toolkit-Lite2（`rknnlite`）在 RK3588 板端运行 `yolo26s-rk3588.rknn` 的模块化推理代码。
当前模型是 Ultralytics 官方导出的 **FP16 版本**（未剪枝、未做 INT8 量化）。

## 模型规格

| 项目 | 值 |
| --- | --- |
| 模型 | `yolo26s_rknn_model/yolo26s-rk3588.rknn` |
| 输入 | `images`，`[1, 3, 640, 640]`，RGB |
| 归一化 | 模型内置 `mean=[0,0,0]`、`std=[255,255,255]`，板端直接喂 **uint8 RGB NHWC** |
| 输出 | `output0`，`[1, 84, 8400]`，float |
| 输出含义 | 前 4 通道 `cx, cy, w, h`（640 输入尺度像素坐标）；后 80 通道类别分数（已 sigmoid） |
| 后处理 | 按行取最大类别 -> 置信度过滤 -> NMS（官方导出是 one2many，需要 NMS） |

## 文件结构

```
yolo26_rknn/                 # 推理库（与示例解耦，可单独拷贝到板上）
├── config.py                # YOLO26Config 配置、COCO 类别兜底
├── metadata.py              # 解析 metadata.yaml（类别名/输入尺寸/任务）
├── preprocessing.py         # letterbox、RGB 转换、坐标还原
├── postprocessing.py        # output0 解码、按类别 NMS
├── backends.py              # RKNNLiteBackend（板端）/ OnnxBackend（PC 验证）
├── detector.py              # YOLO26Detector：串联整个流程，记录分段耗时
├── visualization.py         # 画框
├── results.py               # Detection 数据结构
├── utils.py                 # 中文路径读写、视频源解析
└── cli.py                   # 示例共用参数

detect_image.py              # 单图推理
detect_camera.py             # 摄像头 / 视频 / RTSP 推理
benchmark.py                 # 预处理 / NPU 推理 / 后处理耗时统计
requirements.txt
```

## 板端环境准备

1. 安装与板端 NPU 驱动匹配的 RKNN-Toolkit-Lite2（aarch64 wheel，Python 版本要对应）：

   ```bash
   pip install rknn_toolkit_lite2-<version>-cp<py>-cp<py>-linux_aarch64.whl
   ```

   wheel 下载：<https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages>

2. 安装 Python 依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. 把 `yolo26_rknn/`、`detect_*.py`、`benchmark.py`、`bus.jpg`，以及整个
   `yolo26s_rknn_model/` 目录（`.rknn` + `metadata.yaml`）拷到板上，保持相对路径不变。

## 快速开始

```bash
# 单图
python detect_image.py bus.jpg
python detect_image.py bus.jpg --show --conf 0.3

# 摄像头 0 / USB 摄像头 2 / 视频文件 / RTSP
python detect_camera.py
python detect_camera.py 2 --width 1280 --height 720
python detect_camera.py test.mp4 --save-video result.mp4
python detect_camera.py rtsp://user:pass@192.168.1.10/stream --no-show

# 性能统计（会预热，统计均值/中位数/P95）
python benchmark.py bus.jpg --runs 100
python benchmark.py bus.jpg --runs 100 --json
```

无显示器时 `detect_camera.py` 会自动用 `--no-show`，每 30 帧打印一次 FPS。

### 常用参数

| 参数 | 说明 |
| --- | --- |
| `--conf` | 置信度阈值，默认 0.25 |
| `--iou` | NMS IoU，默认 0.7 |
| `--max-det` | 最多保留框数，默认 300 |
| `--no-nms` | 关闭 NMS（换成 one2one/NMS-free 头导出的模型时使用） |
| `--agnostic-nms` | 跨类别 NMS |
| `--core-mask` | NPU 核：`0` / `1` / `2` / `0_1` / `0_1_2`，默认三核 |
| `--imgsz` | 覆盖输入尺寸，默认读 `metadata.yaml` |
| `--model` | RKNN 模型路径 |
| `--metadata` | metadata 路径，默认模型同目录 |
| `--onnx-model` | PC 上用 ONNX Runtime 验证同一套前后处理（需 `pip install onnxruntime`） |

## 在 PC 上先验证（可选）

没有板子时可以用 ONNX 后端跑完全相同的 letterbox/解码/NMS 流程：

```bash
pip install onnxruntime
python detect_image.py bus.jpg --onnx-model yolo26s.onnx
python benchmark.py bus.jpg --onnx-model yolo26s.onnx --runs 20
```

注意：PC 上 OpenCV 的 `imread` 不支持中文路径，库内已用 `imdecode` 兜底，可直接使用项目路径。

## 作为库调用

```python
from yolo26_rknn import YOLO26Config, YOLO26Detector, draw_detections
from yolo26_rknn.utils import imread, imwrite

config = YOLO26Config(
    model_path="yolo26s_rknn_model/yolo26s-rk3588.rknn",
    conf_thres=0.25,
    iou_thres=0.7,
    core_mask="0_1_2",
)
detector = YOLO26Detector(config)          # 会加载 .rknn 并初始化 NPU
image = imread("bus.jpg")
detections = detector.detect(image)        # 输入 BGR，返回原图坐标系的 Detection
annotated = draw_detections(image, detections)
imwrite("result.jpg", annotated)
print(detector.last_timings)               # preprocess/inference/postprocess ms
detector.release()
```

多路并发时可创建多个检测器，分别绑定不同 NPU 核（`core_mask="0"`、`"1"`、`"2"`）。

## 扩展点

- **换运行时**：实现 `Backend` 协议的 `load / to_model_input / infer / release` 四个方法即可，
  然后 `YOLO26Detector(config, backend=MyBackend())`。
- **换模型**：保持 Ultralytics 单输出 `output0` 格式时，只需替换 `.rknn` 和 `metadata.yaml`；
  类别数变化会通过 metadata 自动适配，输出通道数不匹配时会报明确错误。
- **调阈值**：剪枝/量化后精度会有变化，建议重新扫一遍 `--conf` / `--iou`。

## 后续剪枝与量化的注意事项

1. **剪枝**：在导出 ONNX 前完成。只要类别数和输出布局不变（仍是 `output0: [1, 84, 8400]`），
   本库前后处理无需修改；但权重变化后阈值和精度需要重新验证。
2. **INT8 量化**：
   - 独立使用 rknn-toolkit2 时，保持单输出 `output0`、`do_quantization=True` 并准备校准数据集；
     确认 `rknnlite.inference` 返回的是 float（已反量化）还是 int8。若为 int8，需要在
     `postprocessing.decode_predictions` 前按 `scale/zero_point` 反量化（可在 `decode_predictions`
     前插入，扩展成本很低）。
   - 若使用 Ultralytics 8.4+ 的 `quantize=8` 导出：它会把 NPU 图截断到检测头卷积，
     另外生成 `tail.onnx` 在 CPU 上做框解码（见 `ultralytics/utils/export/rknn.py`）。
     这种模型的输出布局与本库面向的 `output0` 不同，需要额外实现 tail 解码。
   - 当前 FP16 模型无需校准数据集，直接 `rknnlite` 加载即可。

## 参考项目

- ONNX 转 RKNN 及测试脚本：<https://github.com/cqu20160901/yolo26_onnx_rknn>
- RK3588 C++ 板端部署：<https://github.com/cqu20160901/yolo26_rknn_Cplusplus>
- YOLO26 RK3588 部署讲解视频：<https://www.bilibili.com/video/BV1gNr4BFEZP>
- Ultralytics 官方仓库（YOLO26 导出/量化）：<https://github.com/ultralytics/ultralytics>

本目录下的 `yolo26-rk3588-python-master/` 是早期参考代码（其输出为自定义 6 输出头，
与本模型官方单输出 `output0` 不同，后处理不能直接混用）。
