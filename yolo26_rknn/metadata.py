"""解析 Ultralytics 导出目录中的 metadata.yaml（不依赖 PyYAML，板端可直接用）。

只读取推理需要的字段：names、imgsz、task、stride、end2end。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


@dataclass
class ModelMetadata:
    names: Optional[Dict[int, str]] = None
    imgsz: Optional[Tuple[int, int]] = None  # (height, width)
    task: Optional[str] = None
    stride: Optional[int] = None
    end2end: Optional[bool] = None


_IMGSZ_ITEM = re.compile(r"^\s*-\s*(\d+)\s*$")
_NAMES_ITEM = re.compile(r"^\s{2,}(\d+)\s*:\s*(.+?)\s*$")


def parse_metadata_text(text: str) -> ModelMetadata:
    meta = ModelMetadata()
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("task:"):
            meta.task = stripped.split(":", 1)[1].strip().strip("'\"") or None
        elif stripped.startswith("stride:"):
            try:
                meta.stride = int(stripped.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif stripped.startswith("end2end:"):
            meta.end2end = stripped.split(":", 1)[1].strip().lower() == "true"
        elif stripped.startswith("imgsz:"):
            inline = re.findall(r"\d+", stripped.split(":", 1)[1])
            if inline:
                values = inline
            else:
                values = []
                j = i + 1
                while j < len(lines):
                    item = _IMGSZ_ITEM.match(lines[j])
                    if not item:
                        break
                    values.append(item.group(1))
                    j += 1
                i = j - 1
            if len(values) == 1:
                values = values * 2
            if len(values) >= 2:
                meta.imgsz = (int(values[0]), int(values[1]))
        elif stripped.startswith("names:"):
            names: Dict[int, str] = {}
            j = i + 1
            while j < len(lines):
                item = _NAMES_ITEM.match(lines[j])
                if not item:
                    break
                names[int(item.group(1))] = item.group(2).strip("'\"")
                j += 1
            if names:
                meta.names = names
                i = j - 1
        i += 1
    return meta


def load_metadata(path: Union[str, Path]) -> Optional[ModelMetadata]:
    """读取 metadata.yaml；文件不存在或解析失败时返回 None。"""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        return parse_metadata_text(text)
    except Exception:  # 元数据只是辅助信息，解析失败不应阻塞推理
        return None


def names_list(names: Dict[int, str]) -> list:
    """把 {0: 'person', 2: 'car'} 展开为按索引排列的列表。"""
    size = max(names) + 1
    return [names.get(i, str(i)) for i in range(size)]
