"""推理后端封装。

RKNNLiteBackend 用于 RK3588 板端（rknn-toolkit-lite2）；
OnnxBackend 用于 PC 上验证同一套前后处理（需要 onnxruntime，可选）。

后端只接收/返回 numpy，不关心业务逻辑，后续换模型/换运行时只需实现
load / to_model_input / infer / release 四个方法。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Protocol, Union

import numpy as np

_CORE_MASK_ATTRS = {
    "0": "NPU_CORE_0",
    "1": "NPU_CORE_1",
    "2": "NPU_CORE_2",
    "0_1": "NPU_CORE_0_1",
    "0_1_2": "NPU_CORE_0_1_2",
    "auto": "NPU_CORE_AUTO",
}


class Backend(Protocol):
    """所有推理后端需要实现的最小接口。"""

    def load(self) -> None: ...

    def to_model_input(self, image_rgb: np.ndarray) -> np.ndarray: ...

    def infer(self, model_input: np.ndarray) -> List[np.ndarray]: ...

    def release(self) -> None: ...


class RKNNLiteBackend:
    """RK3588 板端 rknn-toolkit-lite2 后端。

    当前 .rknn 模型是用 mean=[0,0,0]、std=[255,255,255] 构建的，
    因此直接把 uint8 RGB NHWC 图像传给 rknnlite 即可，归一化由模型内部完成。
    """

    def __init__(self, model_path: Union[str, Path], core_mask: Union[str, int] = "0_1_2", verbose: bool = False):
        self.model_path = Path(model_path)
        self.core_mask = core_mask
        self.verbose = verbose
        self._rknn = None

    def _resolve_core_mask(self, rknn_lite_cls) -> int:
        if not isinstance(self.core_mask, str):
            return int(self.core_mask)
        attr = _CORE_MASK_ATTRS.get(self.core_mask.strip().lower())
        if attr is None:
            raise ValueError(
                f"不支持的 core_mask={self.core_mask!r}，可选: {sorted(_CORE_MASK_ATTRS)} 或 RKNNLite 常量整数"
            )
        value = getattr(rknn_lite_cls, attr, None)
        if value is None:  # 某些版本没有 NPU_CORE_AUTO
            if self.core_mask.strip().lower() == "auto":
                value = getattr(rknn_lite_cls, "NPU_CORE_0_1_2", None)
        if value is None:
            raise RuntimeError(f"当前 RKNNLite 版本不支持 {attr}")
        return int(value)

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"找不到 RKNN 模型: {self.model_path}")

        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:
            raise RuntimeError(
                "未找到 rknnlite。板端请先安装 RKNN-Toolkit-Lite2，例如:\n"
                "  pip install rknn_toolkit_lite2-<version>-cp<py>-cp<py>-linux_aarch64.whl\n"
                "wheel 与 NPU 驱动版本需匹配 RK3588。"
            ) from exc

        self._rknn = RKNNLite(verbose=self.verbose)
        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            self._rknn = None
            raise RuntimeError(f"rknnlite.load_rknn 失败, ret={ret}, 模型: {self.model_path}")

        ret = self._rknn.init_runtime(core_mask=self._resolve_core_mask(RKNNLite))
        if ret != 0:
            self._rknn.release()
            self._rknn = None
            raise RuntimeError(f"rknnlite.init_runtime 失败, ret={ret}")

    def to_model_input(self, image_rgb: np.ndarray) -> np.ndarray:
        # (H, W, 3) uint8 RGB -> (1, H, W, 3) NHWC uint8
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        return np.ascontiguousarray(image_rgb[None])

    def infer(self, model_input: np.ndarray) -> List[np.ndarray]:
        if self._rknn is None:
            raise RuntimeError("RKNNLite 尚未加载，请先调用 load()")
        outputs = self._rknn.inference(inputs=[model_input])
        if outputs is None:
            raise RuntimeError("rknnlite.inference 返回 None（推理失败）")
        # fp16 构建的模型可能返回 float16，统一转 float32 方便后处理
        return [np.asarray(out, dtype=np.float32) for out in outputs]

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release()
            self._rknn = None

    def __enter__(self) -> "RKNNLiteBackend":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class OnnxBackend:
    """PC 验证用 ONNX Runtime 后端（可选依赖 onnxruntime）。

    与 RKNN 后端的差异：
        RKNN: uint8 RGB, NHWC，归一化在模型里（mean=0, std=255）
        ONNX: float32 RGB, NCHW, [0, 1]，需要在 Python 侧归一化
    """

    def __init__(self, model_path: Union[str, Path], providers: Optional[List[str]] = None):
        self.model_path = Path(model_path)
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._input_name: Optional[str] = None

    def load(self) -> None:
        if not self.model_path.is_file():
            raise FileNotFoundError(f"找不到 ONNX 模型: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("PC 调试需要安装 onnxruntime: pip install onnxruntime") from exc

        self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
        self._input_name = self._session.get_inputs()[0].name

    def to_model_input(self, image_rgb: np.ndarray) -> np.ndarray:
        # (H, W, 3) uint8 RGB -> (1, 3, H, W) float32 [0, 1]
        blob = np.asarray(image_rgb, dtype=np.float32) / 255.0
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])

    def infer(self, model_input: np.ndarray) -> List[np.ndarray]:
        if self._session is None:
            raise RuntimeError("ONNX 模型尚未加载，请先调用 load()")
        outputs = self._session.run(None, {self._input_name: model_input})
        return [np.asarray(out, dtype=np.float32) for out in outputs]

    def release(self) -> None:
        self._session = None

    def __enter__(self) -> "OnnxBackend":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def create_backend(config, backend: str = "rknn") -> Backend:
    """按配置创建后端。backend 可选 "rknn" 或 "onnx"。"""
    if backend == "rknn":
        return RKNNLiteBackend(config.model_path, core_mask=config.core_mask, verbose=False)
    if backend == "onnx":
        return OnnxBackend(config.model_path)
    raise ValueError(f"未知 backend={backend!r}，可选 'rknn' / 'onnx'")
